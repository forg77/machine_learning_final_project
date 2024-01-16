import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

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
        if hc != '0':
          out, hc = self.lstm(x, hc)
        else:
          out, hc = self.lstm(x)
        # print("hidden shape",hc[0].shape,hc[1].shape)
        # print(hc)
        out = self.dropout(out)
        out = self.fc(out[:,-1,:])
        # print("out shape",out.shape)
        
        return out, hc

input_size = len(features)
hidden_size = 2048
num_layers = 1
output_size = len(features)

model = MultiVarLSTM(input_size, hidden_size, num_layers, output_size)
# model = model.to(torch.device("cuda:1" if torch.cuda:1.is_available() else "cpu"))

total_params = sum(p.numel() for p in model.parameters())
print(total_params)
# # 转换为PyTorch的张量
train_X, train_y = seq_tar_spliter(train_set)
con_zeros1 = np.zeros_like(train_X)
# train_X = np.concatenate((train_X,con_zeros1),1)
train_X_tensor = Variable(torch.Tensor(train_X))
train_y_tensor = Variable(torch.Tensor(train_y))

print(train_X.shape)


val_X, val_y = seq_tar_spliter(val_set)
con_zeros1 = np.zeros_like(val_X)
# val_X = np.concatenate((val_X,con_zeros1),1)
val_X_tensor = Variable(torch.Tensor(val_X))
val_y_tensor = Variable(torch.Tensor(val_y))
print(val_X.shape)

test_X, test_y = seq_tar_spliter(test_set)
con_zeros1 = np.zeros_like(test_X)
# test_X = np.concatenate((test_X,con_zeros1),1)
test_X_tensor = Variable(torch.Tensor(test_X)).to(torch.device("cuda:1" if torch.cuda.is_available() else "cpu"))
test_y_tensor = Variable(torch.Tensor(test_y)).to(torch.device("cuda:1" if torch.cuda.is_available() else "cpu"))
print(test_X.shape)
# # 定义损失函数和优化器
criterion = nn.MSELoss()


#切分成小batch

dataset = TensorDataset(train_X_tensor, train_y_tensor)
#*****
batch_size = 64
#******
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
valset = TensorDataset(val_X_tensor, val_y_tensor)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)
testset = TensorDataset(test_X_tensor, test_y_tensor)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)


# checkpoint = torch.load('/home/ycchen/machine_learning/lstm/Final_Rn_Standard_scalar02_336_lstm_model_batch_64_epoch=20_lr_1e-3_layer=1_hid_1024.pth')#1 fail
# checkpoint = torch.load('/home/ycchen/machine_learning/lstm/Final_Rn_Standard_scalar02_336_lstm_model_batch_64_epoch=20_lr_5e-3_layer=1_hid_1024.pth') #2 fail
# checkpoint = torch.load('/home/ycchen/machine_learning/lstm/Final_Rn_Standard_scalar02_336_lstm_model_batch_64_epoch=20_lr_1e-2_layer=1_hid_1024.pth') #3
# checkpoint = torch.load('/home/ycchen/machine_learning/lstm/11_Final_Rn_Standard_scalar02_336_lstm_model_batch_256_epoch=20_lr_1e-1_layer=1_hid_1024.pth') #4
checkpoint = torch.load('/home/ycchen/machine_learning/lstm/14_Final_Rn_Standard_scalar02_336_lstm_model_batch_256_epoch=50_lr_1e-4_layer=1_hid_2048.pth') #5

model = model.to(torch.device("cuda:1" if torch.cuda.is_available() else "cpu"))
model.load_state_dict(checkpoint)

#测试
model.eval()  # 设置模型为评估模式
draw_seq_pre = []
draw_seq_tar = []
test_loss1 = 0.0
test_loss2 = 0.0
predictions = []
diffsum = 0
criterion1 = nn.MSELoss() #MSE
criterion2 = nn.L1Loss() #MAE
MSE_test_ave_loss = []
MAE_test_ave_loss = []
drawn = 0
for check in range(5):
  print(check)
  test_loss1 = 0.0
  test_loss2 = 0.0
  total_add_1 = 0.0
  total_add_2 = 0.0
  with torch.no_grad():
    for test_batch_data, test_batch_labels in testloader:
        # test_batch_data = test_batch_data.to("cuda:1")
        # test_batch_labels = test_batch_labels.to("cuda:1")
        test_outputs, hc = model(test_batch_data,'0')
        
        test_outputs = test_outputs.unsqueeze(1)
        if drawn == 3:
          draw_seq_pre.append(test_outputs[9,-1,-1].to("cpu"))
          draw_seq_tar.append(test_batch_labels[9,0,-1].to("cpu"))
        total_add_1 += criterion1(test_outputs[:,-1,:], test_batch_labels[:,0,:])
        total_add_2 += criterion2(test_outputs[:,-1,:], test_batch_labels[:,0,:])
        outputs = test_outputs
        for i in range(1,336):
          outputs, hc = model(outputs, hc)
          outputs1 = outputs.unsqueeze(1)
          outputs = outputs1
          if drawn == 3:
            draw_seq_pre.append(outputs1[9,-1,-1].to("cpu"))
            draw_seq_tar.append(test_batch_labels[9,i,-1].to("cpu"))
          # hc = (torch.squeeze(hc[0]),torch.squeeze(hc[1]))
          total_add_1 += criterion1(outputs1[:,-1,:], test_batch_labels[:,i,:]).item()
          total_add_2 += criterion2(outputs1[:,-1,:], test_batch_labels[:,i,:]).item()
        total_add_2 /= 336
        total_add_1 /= 336
        test_loss1 += total_add_1.item()
        test_loss2 += total_add_2.item()
        drawn += 1
  MSE_test_ave_loss.append(test_loss1 / len(testloader))
  MAE_test_ave_loss.append(test_loss2 / len(testloader))
array1 = np.array(MSE_test_ave_loss)
array2 = np.array(MAE_test_ave_loss)
print("MSE:",np.mean(array1),"MAE:",np.mean(array2))
print("MSEstd",np.std(array1),"MAEstd",np.std(array2))
# /home/ycchen/machine_learning/lstm/Final_Rn_Standard_scalar02_336_lstm_model_batch_64_lr_1e-2_layer=4_hid_256.pth
# MSE: 1.1882322183797056 MAE: 0.8460553243686336
# MSEstd 0.00011130385292468906 MAEstd 4.232856445658689e-05

#折线图绘制
plt.plot(draw_seq_pre, label='prediction', linestyle='-',marker='o',markersize=1)
plt.plot(draw_seq_tar, label='ground truth', linestyle='--',marker='s',markersize=1)

# 添加标签和标题
plt.xlabel('Time Step')
plt.ylabel('Values')
plt.title('Two Sequences Comparison')

# 添加图例
plt.legend()

plt.savefig('./lstm/lstm_336_1024x1_comparison_plot_6.png')
