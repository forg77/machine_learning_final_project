import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import time
torch.manual_seed(114514)
FILE_PATH = '/home/ycchen/machine_learning/ETTh1.csv'
WINDOW_SIZE = 96 + 336
TOTAL_ROWS = 17420
FEATURE_NUM = 7
df = pd.read_csv(FILE_PATH)

#I 96+ O 336


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
#data_scaled[1] = [ 0.22541532 -0.08110291  0.25291664 -0.07024342  0.10055753  0.24156671 0.27246594]
# len(data_scaled) = 17420
for i in range(TOTAL_ROWS-WINDOW_SIZE):
  now_list = data_scaled[i:i+WINDOW_SIZE]
  data.append(now_list)
# print(len(data))


TOTAL_SIZE_96_336 = 16988
TRAIN_SIZE = int(0.6 * TOTAL_SIZE_96_336)
VAL_SIZE = int(0.2 * TOTAL_SIZE_96_336)
TEST_SIZE = TOTAL_SIZE_96_336 - TRAIN_SIZE - VAL_SIZE
train_set = data[:TRAIN_SIZE]
val_set = data[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
test_set = data[-TEST_SIZE:]
# # print(len(val_set),len(val_set[0]))
# # 3345 192

def seq_tar_spliter(input_list,window_size = WINDOW_SIZE):

  input_list = np.array(input_list)
  window_size = 96
  seq = input_list[:,:window_size,:]
  label = input_list[:,window_size:,:]
  return np.array(seq), np.array(label)

# seq, label = spliter(val_set)
# # print(seq.shape)
# # print(label.shape)
# # (3445, 96, 7)
# # (3445, 96, 7)


class MultiVarLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MultiVarLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, hc):
        if hc == '0':
          out, hc = self.lstm(x)
        else:
          out, hc = self.lstm(x, hc)
        # print("hidden shape",hc[0].shape,hc[1].shape)
        # print(hc)
        out = self.fc(out[:,-1,:])
        out = self.dropout(out)
        # print("out shape",out.shape)
        
        return out, hc

input_size = len(features)
hidden_size = 2048
num_layers = 1
output_size = len(features)

model = MultiVarLSTM(input_size, hidden_size, num_layers, output_size)
model = model.to(torch.device("cuda:4" if torch.cuda.is_available() else "cpu"))


# # 转换为PyTorch的张量
train_X, train_y = seq_tar_spliter(train_set)
# print(train_X.shape,train_y.shape)
con_zeros1 = np.zeros_like(train_X)
# train_X = np.concatenate((train_X,con_zeros1),1)
train_X_tensor = Variable(torch.Tensor(train_X)).to("cuda:4")
train_y_tensor = Variable(torch.Tensor(train_y)).to("cuda:4")

print(train_X.shape)


val_X, val_y = seq_tar_spliter(val_set)
con_zeros1 = np.zeros_like(val_X)
# val_X = np.concatenate((val_X,con_zeros1),1)
val_X_tensor = Variable(torch.Tensor(val_X)).to("cuda:4")
val_y_tensor = Variable(torch.Tensor(val_y)).to("cuda:4")
print(val_X.shape)

test_X, test_y = seq_tar_spliter(test_set)
con_zeros1 = np.zeros_like(test_X)
# test_X = np.concatenate((test_X,con_zeros1),1)
test_X_tensor = Variable(torch.Tensor(test_X)).to("cuda:4")
test_y_tensor = Variable(torch.Tensor(test_y)).to("cuda:4")
print(test_X.shape)
# # 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

#切分成小batch

dataset = TensorDataset(train_X_tensor, train_y_tensor)
#*****
batch_size = 256
#******
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
valset = TensorDataset(val_X_tensor, val_y_tensor)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)
testset = TensorDataset(test_X_tensor, test_y_tensor)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
# # 训练模型
num_epochs = 10
STOP_SIGN = 336
nowbest = 1.0
for epoch in range(num_epochs):
    t1 = time.time()
    model.train()
    for batch_seq, batch_target in dataloader:
      batch_seq, batch_target = batch_seq.to("cuda:4"), batch_target.to("cuda:4")
      output,hc = model(batch_seq,'0') #这是第97个值，第一个预测值
      # output = output[:,-1,:]
      # print("output",output.shape)
      # hc1 = (torch.squeeze(hc[0]),torch.squeeze(hc[1]))
      # print(hc1[0].shape)
      loss = 0.0
      loss += criterion(output, batch_target[:,0,:])
      outputs = output.unsqueeze(1)
#       print("output shape",outputs.shape)
#       print("target shape",batch_target.shape)
#       output shape torch.Size([16, 1, 7])
#       target shape torch.Size([16, 336, 7])
      # for j in range(16): 
      for i in range(1,STOP_SIGN):
        outputs, hc = model(outputs, hc)
        outputs = outputs.unsqueeze(1)
        # hc = (torch.squeeze(hc[0]),torch.squeeze(hc[1]))
        loss += criterion(outputs[:,-1,:], batch_target[:,i,:])
      
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.7)
      optimizer.step()
    model.eval()  # 设置模型为评估模式
    val_loss = 0.0
    val_total_loss = 0.0
    with torch.no_grad():
        for val_batch_data, val_batch_labels in val_loader:
            val_batch_data, val_batch_labels = val_batch_data.to("cuda:4"), val_batch_labels.to("cuda:4")
            val_outputs, hc = model(val_batch_data,'0')
            val_outputs = val_outputs.unsqueeze(1)
            val_loss += criterion(val_outputs[:,-1,:], val_batch_labels[:,0,:])
            outputs = val_outputs
            for i in range(1,STOP_SIGN):
              outputs, hc = model(outputs, hc)
              outputs = outputs.unsqueeze(1)
              # hc = (torch.squeeze(hc[0]),torch.squeeze(hc[1]))
              val_loss += criterion(outputs[:,-1,:], val_batch_labels[:,i,:])
            val_loss /= 336
            val_total_loss += val_loss.item()
    average_val_loss = val_total_loss / len(val_loader)
    if average_val_loss < 0.9:
        if average_val_loss < nowbest:
            torch.save(model.state_dict(), './lstm/12_Final_Rn_earlystop_Standard_scalar02_336_lstm_model_batch_256_epoch=10_lr_1e-4_layer=1_hid_2048.pth')
            nowbest = average_val_loss
    t2 = time.time()
    print("time cost",t2-t1)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item()/336:.4f}, Validation Loss: {average_val_loss:.4f}')
torch.save(model.state_dict(), './lstm/12_Final_Rn_Standard_scalar02_336_lstm_model_batch_256_epoch=10_lr_1e-4_layer=1_hid_2048.pth')

