import pandas as pd
FILE_PATH = '/home/ycchen/machine_learning/ETTh1.csv'
df = pd.read_csv(FILE_PATH)

#I 96+ O 96
from sklearn.preprocessing import MinMaxScaler,StandardScaler
WINDOW_SIZE = 96 + 96
TOTAL_ROWS = 17420
FEATURE_NUM = 7
features = ['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
# scaler = MinMaxScaler(feature_range=(-1, 1))
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
#data_scaled[1] = [ 0.22541532 -0.08110291  0.25291664 -0.07024342  0.10055753  0.24156671 0.27246594]
# len(data_scaled) = 17420
for i in range(TOTAL_ROWS-WINDOW_SIZE):
  now_list = data_scaled[i:i+WINDOW_SIZE]
  data.append(now_list)
# len(data) # 17228
# len(data[1]) # 192
# len(data[1][1]) # 7

import numpy as np
TOTAL_SIZE_96 = 17228
TRAIN_SIZE = int(0.6 * TOTAL_SIZE_96)
VAL_SIZE = int(0.2 * TOTAL_SIZE_96)
TEST_SIZE = TOTAL_SIZE_96 - TRAIN_SIZE - VAL_SIZE
train_set = data[:TRAIN_SIZE]
val_set = data[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
test_set = data[-TEST_SIZE:]
# print(len(val_set),len(val_set[0]))
# 3345 192

def seq_tar_spliter(input_list,window_size = WINDOW_SIZE):

  input_list = np.array(input_list)
  window_size = int(window_size/2)
  seq = input_list[:,:window_size,:]
  label = input_list[:,window_size:,:]
  return np.array(seq), np.array(label)

# seq, label = spliter(val_set)
# print(seq.shape)
# print(label.shape)
# (3445, 96, 7)
# (3445, 96, 7)

import torch
import torch.nn as nn
from torch.autograd import Variable
torch.manual_seed(114514)
class MultiVarLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MultiVarLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hc):
        if hc != '0':
          out, hc = self.lstm(x, hc)
        else:
          out, hc = self.lstm(x)
        # print("hidden shape",hc[0].shape,hc[1].shape)
        # print(hc)
        out = self.fc(out[:,-1,:])
        # print("out shape",out.shape)
        
        return out, hc

input_size = len(features)
hidden_size = 256
num_layers = 4
output_size = len(features)

model = MultiVarLSTM(input_size, hidden_size, num_layers, output_size)
model = model.to(torch.device("cuda:1" if torch.cuda.is_available() else "cpu"))


# 转换为PyTorch的张量
train_X, train_y = seq_tar_spliter(train_set)
con_zeros1 = np.zeros_like(train_X)
# train_X = np.concatenate((train_X,con_zeros1),1)
train_X_tensor = Variable(torch.Tensor(train_X)).to("cuda:1")
train_y_tensor = Variable(torch.Tensor(train_y)).to("cuda:1")

print(train_X.shape)


val_X, val_y = seq_tar_spliter(val_set)
con_zeros1 = np.zeros_like(val_X)
# val_X = np.concatenate((val_X,con_zeros1),1)
val_X_tensor = Variable(torch.Tensor(val_X)).to("cuda:1")
val_y_tensor = Variable(torch.Tensor(val_y)).to("cuda:1")
print(val_X.shape)

test_X, test_y = seq_tar_spliter(test_set)
con_zeros1 = np.zeros_like(test_X)
# test_X = np.concatenate((test_X,con_zeros1),1)
test_X_tensor = Variable(torch.Tensor(test_X)).to("cuda:1")
test_y_tensor = Variable(torch.Tensor(test_y)).to("cuda:1")
print(test_X.shape)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

#切分成小batch
from torch.utils.data import DataLoader, TensorDataset
import time
dataset = TensorDataset(train_X_tensor, train_y_tensor)

batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
valset = TensorDataset(val_X_tensor, val_y_tensor)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)
testset = TensorDataset(test_X_tensor, test_y_tensor)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
# 训练模型
num_epochs = 50
nowbest = 1.0
for epoch in range(num_epochs):
    model.train()
    t1 = time.time()
    for batch_seq, batch_target in dataloader:
      batch_seq, batch_target = batch_seq.to("cuda:1"), batch_target.to("cuda:1")
      output,hc = model(batch_seq,'0') #这是第97个值，第一个预测值
      # output = output[:,-1,:]
      # print("output",output.shape)
      # hc1 = (torch.squeeze(hc[0]),torch.squeeze(hc[1]))
      # print(hc1[0].shape)
      loss = 0.0
      output = output.unsqueeze(1)
      # print(output.shape)
      loss += criterion(output[:,-1,:], batch_target[:,0,:])
      outputs = output
      # for j in range(16): 
      for i in range(1,96):
        outputs, hc = model(outputs, hc)
        outputs = outputs.unsqueeze(1)
        # hc = (torch.squeeze(hc[0]),torch.squeeze(hc[1]))
        loss += criterion(outputs[:,-1,:], batch_target[:,i,:])
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    model.eval()  # 设置模型为评估模式
    val_total_loss = 0.0
    val_loss = 0.0
    
    with torch.no_grad():
        for val_batch_data, val_batch_labels in val_loader:
            val_batch_data, val_batch_labels = val_batch_data.to("cuda:1"), val_batch_labels.to("cuda:1")
            val_outputs, hc = model(val_batch_data,'0')
            val_outputs = val_outputs.unsqueeze(1)
            val_loss += criterion(val_outputs[:,-1,:], val_batch_labels[:,0,:])
            outputs = val_outputs
            for i in range(1,96):
              outputs, hc = model(outputs, hc)
              outputs = outputs.unsqueeze(1)
              # hc = (torch.squeeze(hc[0]),torch.squeeze(hc[1]))
              val_loss += criterion(outputs[:,-1,:], val_batch_labels[:,i,:])
            val_loss /= 96
            val_total_loss += val_loss
    average_val_loss =  val_total_loss / len(val_loader)
    if average_val_loss < 0.75:
       if average_val_loss < nowbest:
        torch.save(model.state_dict(), './lstm/Final_Rn_earlystop_Standard_scalar01_lstm_model_batch_64_lr_1e-2_layer=4_hid_256.pth')
        nowbest = average_val_loss
    t2 = time.time()
    print("time cost",t2-t1)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item()/96:.4f}, Validation Loss: {average_val_loss:.4f}')
torch.save(model.state_dict(), './lstm/Final_Rn_Standard_scalar01_lstm_model_batch_64_lr_1e-2_layer=4_hid_256.pth')

