from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import torch
from torch import nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

class MultiVarTransformer(nn.Module):
    def __init__(self, input_size, output_size, num_encoder_layers, num_decoder_layers,batch_first=True):
        super(MultiVarTransformer, self).__init__()
        self.d_model = 128
        self.batch_first = batch_first
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8,)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=8)
        self.decoder = nn.TransformerDecoder(self.decoder_layer,num_layers=num_decoder_layers) 
        self.fc1 = nn.Linear(input_size,self.d_model)
        self.fc2 = nn.Linear(output_size,self.d_model)
        self.fc3 = nn.Linear(self.d_model,output_size)
        self.transformer = nn.Transformer(d_model=self.d_model,
                                          custom_encoder=self.encoder,
                                          custom_decoder=self.decoder,
                                          batch_first=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self, src, tgt):
        if torch.isnan(src).any():
            print("ERROR at src input!")
        if torch.isnan(tgt).any():
            print("ERROR at tgt input!")
        src = self.fc1(src)
        src = self.relu(src)
        if torch.isnan(src).any():
            print("ERROR at fc1 output!")
        tgt = self.fc2(tgt)
        tgt = self.relu(tgt)
        if torch.isnan(tgt).any():
            print("ERROR at fc2 output!")
        # print(src.shape,tgt.shape)
        output = self.transformer(src, tgt)
        output = self.dropout(output)
        if torch.isnan(output).any():
            print("ERROR at fc3 input!")
            opt1 = self.encoder(src)
            if torch.isnan(opt1).any():
                print("ERROR at encoder output!")
            else:
                print("ERROR at decoder output!")

        output = self.fc3(output)

        if torch.isnan(output).any():
            print("ERROR at final output!")
        return output
        
            

FILE_PATH = '/home/ycchen/machine_learning/ETTh1.csv'
df = pd.read_csv(FILE_PATH)
#I 96+ O 336
#HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
upper_bound_336 = 95 + 336
upper_bound_96 = 95 + 96
#数据预处理
WINDOW_SIZE = 96 + 336
TOTAL_ROWS = 17420
FEATURE_NUM = 7
features = ['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df[features])
HUFL = df['HUFL']
HULL = df['HULL']
MUFL = df['MUFL']
MULL = df['MULL']
LUFL = df['LUFL']
LULL = df['LULL']
OT = df['OT']
data = []

for i in range(TOTAL_ROWS-WINDOW_SIZE):
  now_list = data_scaled[i:i+WINDOW_SIZE]
  data.append(now_list)
print(len(data))
TOTAL_SIZE_336 = len(data)
TRAIN_SIZE = int(0.6 * TOTAL_SIZE_336)
VAL_SIZE = int(0.2 * TOTAL_SIZE_336)
TEST_SIZE = TOTAL_SIZE_336 - TRAIN_SIZE - VAL_SIZE
print(TEST_SIZE)
train_set = data[:TRAIN_SIZE]
val_set = data[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
test_set = data[-TEST_SIZE:]
def seq_tar_spliter(input_list,window_size = WINDOW_SIZE):

  input_list = np.array(input_list)
  window_size = 96
  seq = input_list[:,:,:upper_bound_336]
  label = input_list[:,:,96:]
  return np.array(seq), np.array(label)
#模型
exp = MultiVarTransformer(input_size=96, output_size=336, num_encoder_layers=3, num_decoder_layers=3)
exp = exp.to(torch.device("cuda:5" if torch.cuda.is_available() else "cpu"))
total_params = sum(p.numel() for p in exp.parameters())
print(total_params)
from torch.autograd import Variable
train_set = Variable(torch.tensor(np.array(train_set)).to(torch.float32))
train_set = train_set.permute(0,2,1)
print(train_set.shape)
train_size = train_set.shape[0]
val_set = Variable(torch.tensor(np.array(val_set)).to(torch.float32))
val_set = val_set.permute(0,2,1)
print(val_set.shape)
val_size = val_set.shape[0]
test_set = Variable(torch.tensor(np.array(test_set)).to(torch.float32))
test_set = test_set.permute(0,2,1)
test_X, test_y = seq_tar_spliter(test_set)
test_X_tensor = Variable(torch.Tensor(test_X)).to("cuda:5")
test_y_tensor = Variable(torch.Tensor(test_y)).to("cuda:5")
testset = TensorDataset(test_X_tensor, test_y_tensor)
print(test_set.shape)
test_size = test_set.shape[0]
criterion1 = nn.MSELoss()
criterion2 = nn.L1Loss()
#训练
import time
from torch.nn.utils import clip_grad_value_
EPOCH = 300
BATCH_SIZE = 64
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

# checkpoint = torch.load('/home/ycchen/machine_learning/transformer_model_dim_128_336_batch_32_lr_1e-5_epoch=50.pth')
# checkpoint = torch.load('Final_Rn_earlystop_transformer_model_dim_128_336_batch_32_lr_5e-5_epoch=200.pth') #1 
# MSE: 2.058041273118612 MAE: 2.896324363473487
# MSEstd 0.0021574875232783126 MAEstd 0.001004250550855336
checkpoint = torch.load('/home/ycchen/machine_learning/Final_Rn_earlystop_transformer_model_dim_128_336_batch_64_lr_1e-5_epoch=50.pth') #2
exp.load_state_dict(checkpoint)
exp.eval()
MSE_test_ave_loss = []
MAE_test_ave_loss = []
draw_seq_pre = []
draw_seq_tar = []
drawn = 0
for check in range(5):
    print(check)
    batch_count = 0
    test_total_loss_1 = 0.0
    test_total_loss_2 = 0.0
    test_loss_1 = 0.0
    test_loss_2 = 0.0
    with torch.no_grad():
        for test_batch_data, test_batch_labels in testloader:
            test_batch_data, test_batch_labels = test_batch_data.to("cuda:5"), test_batch_labels.to("cuda:5")
         
            upper = min(test_size,i+BATCH_SIZE)
            batch_count += 1
            test_input_src = test_batch_data[:,:,:96]
            test_input_tgt = test_batch_data[:,:,95:upper_bound_336]
            test_set_fin = test_batch_labels
            opt = exp(test_input_src,test_input_tgt)
            # print(opt.shape,val_set_fin.shape)

            for j in range(0,336):
                test_loss_1 += criterion1(opt[:,:,j],test_set_fin[:,:,j]).item()
                test_loss_2 += criterion2(opt[:,:,j],test_set_fin[:,:,j]).item()
                if drawn == 3:
                    draw_seq_pre.append(opt[2,-1,j].item())
                    draw_seq_tar.append(test_set_fin[2,-1,j].item())
            test_loss_1 /= 336
            test_total_loss_1 += test_loss_1
            test_loss_2 /= 336
            test_total_loss_2 += test_loss_2
            drawn += 1
        # print(batch_count)
    MSE_test_ave_loss.append(test_total_loss_1 / len(testloader))
    MAE_test_ave_loss.append(test_total_loss_2 / len(testloader))
array1 = np.array(MSE_test_ave_loss)
array2 = np.array(MAE_test_ave_loss)
print("MSE:",7*np.mean(array1),"MAE:",7*np.mean(array2))
print("MSEstd",7*np.std(array1),"MAEstd",7*np.std(array2))
#5108304
#/home/ycchen/machine_learning/transformer_model_dim_128_336_batch_32_lr_1e-5_epoch=50.pth
#MSE: 0.7129357251482951 MAE: 0.5886753493911224
#/home/ycchen/machine_learning/transformer_model_dim_128_336_batch_32_lr_1e-5_epoch=300.pth
#MSE: 0.6565874784205441 MAE: 0.5373961901720701
#/home/ycchen/machine_learning/Final_Rn_transformer_model_dim_128_336_batch_32_lr_1e-5_epoch=50.pth
#MSE: 0.7072047969941183 MAE: 0.5831019268264193
# MSEstd 0.0 MAEstd 0.0
# /home/ycchen/machine_learning/Final_Rn_transformer_model_dim_128_336_batch_32_lr_5e-5_epoch=20.pth
plt.plot(draw_seq_pre, label='prediction', linestyle='-',marker='o',markersize=1)
plt.plot(draw_seq_tar, label='ground truth', linestyle='--',marker='s',markersize=1)
# 添加标签和标题
plt.xlabel('Time Step')
plt.ylabel('Values')
plt.title('Two Sequences Comparison')

# 添加图例
plt.legend()

plt.savefig('./transformer_336_comparison_plot_2.png')