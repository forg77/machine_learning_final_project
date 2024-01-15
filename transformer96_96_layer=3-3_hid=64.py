from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import torch
from torch import nn as nn
torch.manual_seed(114514)
class MultiVarTransformer(nn.Module):
    def __init__(self, input_size, output_size, num_encoder_layers, num_decoder_layers):
        super(MultiVarTransformer, self).__init__()
        self.d_model = 64
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
        output = self.relu(output)
        if torch.isnan(output).any():
            print("ERROR at final output!")
        return output
        
            

FILE_PATH = '/home/ycchen/machine_learning/ETTh1.csv'
df = pd.read_csv(FILE_PATH)
#I 96+ O 96

WINDOW_SIZE = 96 + 96
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
TOTAL_SIZE_96 = 17228
TRAIN_SIZE = int(0.6 * TOTAL_SIZE_96)
VAL_SIZE = int(0.2 * TOTAL_SIZE_96)
TEST_SIZE = TOTAL_SIZE_96 - TRAIN_SIZE - VAL_SIZE
print(TEST_SIZE)
train_set = data[:TRAIN_SIZE]
val_set = data[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
test_set = data[-TEST_SIZE:]
#模型
exp = MultiVarTransformer(input_size=96, output_size=96, num_encoder_layers=3, num_decoder_layers=3)
exp = exp.to(torch.device("cuda:7" if torch.cuda.is_available() else "cpu"))
import numpy as np
from torch.autograd import Variable
train_set = Variable(torch.tensor(np.array(train_set)).to(torch.float32)).to(torch.device("cuda:7"))
train_set = train_set.permute(0,2,1)
print(train_set.shape)
train_size = train_set.shape[0]
val_set = Variable(torch.tensor(np.array(val_set)).to(torch.float32)).to(torch.device("cuda:7"))
val_set = val_set.permute(0,2,1)
print(val_set.shape)
val_size = val_set.shape[0]
test_set = Variable(torch.tensor(np.array(test_set)).to(torch.float32)).to(torch.device("cuda:7"))
test_set = test_set.permute(0,2,1)
print(test_set.shape)
test_size = test_set.shape[0]
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(exp.parameters(), lr=1e-5, weight_decay=1e-4)
#训练
import time
from torch.nn.utils import clip_grad_value_
EPOCH = 50
BATCH_SIZE = 32
nowbest = 1.0
for _ in range(EPOCH):
    time1 = time.time()
    exp.train()
    for i in range(0,train_size,BATCH_SIZE):
        upper = min(train_size,i+BATCH_SIZE)
        train_input_src = train_set[i:upper,:,:96]
        train_input_tgt = train_set[i:upper,:,95:191]
        train_set_fin = train_set[i:upper,:,96:192]
        opt = exp(train_input_src, train_input_tgt)
        

        loss = 0.0
        for j in range(0,96):
            # if torch.nan(criterion(opt[:,j,:],train_set_fin[:,j,:])):
            #     print("false")

            loss += criterion(opt[:,:,j],train_set_fin[:,:,j])
        loss /= 96
        optimizer.zero_grad()
        loss.backward()
        max_grad_norm = 1.0
        clip_grad_value_(exp.parameters(),max_grad_norm)#防止输出nan
        optimizer.step()
    print("start to eval!")
    exp.eval()
    batch_count = 0
    val_total_loss = 0.0
    val_loss = 0.0
    with torch.no_grad():
        for i in range(0,val_size,BATCH_SIZE):
            upper = min(val_size,i+BATCH_SIZE)
            batch_count += 1
            val_input_src = val_set[i:upper,:,:96]
            val_input_tgt = val_set[i:upper,:,95:191]
            val_set_fin = val_set[i:upper,:,96:192]
            opt = exp(val_input_src,val_input_tgt)
            # print(opt.shape,val_set_fin.shape)
           
            for j in range(0,96):
                val_loss += criterion(opt[:,:,j],val_set_fin[:,:,j])
            val_loss /= 96
            val_total_loss += val_loss

    # print(batch_count)
    val_ave_loss = val_total_loss / batch_count
    if val_ave_loss < 0.65:
        if val_ave_loss < nowbest:
            nowbest = val_ave_loss
            torch.save(exp.state_dict(), 'Final_Rn_earlystop_transformer_model_dim_64_96_batch_32_lr_1e-5_epoch=50.pth')

    if torch.isnan(val_ave_loss):
        print(torch.isnan(opt).any())
    time2 = time.time()
    print("time cost",time2-time1)
    print(f'Epoch [{_+1}/{EPOCH}], Train Loss: {loss.item():.4f}, Validation Loss: {val_ave_loss:.4f}')
torch.save(exp.state_dict(), 'Final_Rn_transformer_model_dim_64_96_batch_32_lr_1e-5_epoch=50.pth')

