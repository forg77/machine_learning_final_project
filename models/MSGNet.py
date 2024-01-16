import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torch.cuda import device

from layers.Embed import DataEmbedding
from layers.MSGBlock import GraphBlock, simpleVIT, Attention_Block, Predict


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]



def STFT_for_Period(x, k=2, n_fft=96, window_size=96, hop_size=48):
    # [B, T, C]
    B, T, C = x.shape
    print(f"Shape of x before padding: {x.shape}")

    total_pad_size = int(max(window_size - T, 0))
    print(f"Total padding size: {total_pad_size}")
    x_padded = F.pad(x, (0, total_pad_size), mode='constant', value=0)


    window_size = int(window_size)
    print(f"window_size: {window_size}, n_fft: {n_fft}, hop_size: {hop_size}")

    # 进行 STFT
    x_padded = x_padded.to(device)  # 将输入移到 GPU 上
    stft_result = torch.stft(x_padded, n_fft=n_fft, window=torch.hann_window(window_size).to(device),
                             hop_length=hop_size, center=True, normalized=False)

    # 取实部和虚部的绝对值
    magnitude = torch.abs(stft_result)

    # 计算频率分量的平均值
    frequency_list = magnitude.mean(0).mean(-1)
    frequency_list[0] = 0  # 将直流分量置零（直流分量对应频率为0）

    # 选择前k个最高频率分量的索引
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()

    # 计算周期
    period = T // top_list

    # 返回周期和对应频率分量的平均值
    return period, magnitude[:, top_list]


class ScaleGraphBlock(nn.Module):
    def __init__(self, configs):
        super(ScaleGraphBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k

        self.att0 = Attention_Block(configs.d_model, configs.d_ff,
                                   n_heads=configs.n_heads, dropout=configs.dropout, activation="gelu")
        self.norm = nn.LayerNorm(configs.d_model)
        self.gelu = nn.GELU()
        self.gconv = nn.ModuleList()
        for i in range(self.k):
            self.gconv.append(
                GraphBlock(configs.c_out , configs.d_model , configs.conv_channel, configs.skip_channel,
                        configs.gcn_depth , configs.dropout, configs.propalpha ,configs.seq_len,
                           configs.node_dim))


    def forward(self, x):
        B, T, N = x.size()

        # scale_list, scale_weight = FFT_for_Period(x, self.k)
        scale_list, scale_weight = STFT_for_Period(x, self.k, self.seq_len, self.pred_len/2)

        res = []
        for i in range(self.k):
            scale = scale_list[i] # fft
            #Gconv
            x = self.gconv[i](x)
            # paddng
            if (self.seq_len) % scale != 0:
                length = (((self.seq_len) // scale) + 1) * scale
                padding = torch.zeros([x.shape[0], (length - (self.seq_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            out = out.reshape(B, length // scale, scale, N)

        #for Mul-attetion
            out = out.reshape(-1 , scale , N)
            out = self.norm(self.att0(out))
            out = self.gelu(out)
            out = out.reshape(B, -1 , scale , N).reshape(B ,-1 ,N)
        # #for simpleVIT
        #     out = self.att(out.permute(0, 3, 1, 2).contiguous()) #return
        #     out = out.permute(0, 2, 3, 1).reshape(B, -1 ,N)

            out = out[:, :self.seq_len, :]
            res.append(out)

        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        scale_weight = F.softmax(scale_weight, dim=1)
        scale_weight = scale_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * scale_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # for graph
        # self.num_nodes = configs.c_out
        # self.subgraph_size = configs.subgraph_size
        # self.node_dim = configs.node_dim
        # to return adj (node , node)
        # self.graph = constructor_graph()

        self.model = nn.ModuleList([ScaleGraphBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model,
                                           configs.embed, configs.freq, configs.dropout)

        # lstm
        # self.lstm = nn.LSTM(input_size=configs.d_model, hidden_size=256,
        #                     num_layers=1, batch_first=True)
        # 1D CNN
        self.conv1d = nn.Conv1d(in_channels=configs.d_model, out_channels=16, kernel_size=3, padding=1)

        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.predict_linear = nn.Linear(
            self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(
            configs.d_model, configs.c_out, bias=True)
        self.seq2pred = Predict(configs.individual ,configs.c_out,
                                configs.seq_len, configs.pred_len, configs.dropout)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # print(enc_out.shape)
        # adp = self.graph(torch.arange(self.num_nodes).to(self.device))

        # print(enc_out.shape)

        for i in range(self.layer):
            # 将 embedding 输出传递给 1D 卷积层
            enc_out = self.conv1d(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
            # ScaleGraphBlock
            enc_out = self.layer_norm(self.model[i](enc_out))

        # porject back
        dec_out = self.projection(enc_out)
        dec_out = self.seq2pred(dec_out.transpose(1, 2)).transpose(1, 2)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len, 1))

        return dec_out[:, -self.pred_len:, :]