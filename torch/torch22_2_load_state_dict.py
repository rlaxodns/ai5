import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import tqdm

from sklearn.metrics import r2_score
from torch.utils.data import DataLoader # batch 정의
from torch.utils.data.dataset import Dataset


random.seed(333)
np.random.seed(333)
torch.manual_seed(333) # torch 고정
torch.cuda.manual_seed(333) # cuda 고정

# RNN과 LSTM의 차이는 c0(cell state)가 추가된 것 뿐이다

# USE_CUDA = torch.cuda.is_available()
# DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')

# print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(DEVICE)

# 판다스만 열/행 나머지는 행/열
# 판다스를 행/열로 사용하려면 loc, iloc
# iloc : index location -> i는 index이지만 int로 외워라
# loc : 

PATH = './_data/kaggle/netflix/'

train_csv = pd.read_csv(PATH + 'train.csv')

print(train_csv)
train_csv.info()
print(train_csv.isna().sum())
print(train_csv.describe())

data = train_csv.iloc[:, 1:4]

data['종가'] = train_csv['Close']

# print(data)
# hist = data.hist()
# plt.show()

# 아래코드는 컬럼 구분없이 scale되는 문제가 있음
# data = train_csv.iloc[:, 1:4].values
# data = (data - np.min(data)) / (np.max(data) - np.min(data))
# data = pd.DataFrame(data)

# 아래코드로 수정
# data = (data - np.min(data, axis = 0)) / (np.max(data, axis = 0) - np.min(data, axis = 0))

# print(data.describe())

class Custom_Dataset(Dataset):
    def __init__(self, train_csv : pd.DataFrame):
        self.csv = train_csv

        self.x = self.csv.iloc[:, 1:4].values
        
        # 정규화
        self.x = (self.x - np.min(self.x, axis = 0)) / (np.max(self.x, axis = 0) - np.min(self.x, axis = 0))

        self.y = self.csv['Close']

    def __len__(self):
        return len(self.x) - 30

    def __getitem__(self, i):
        x = self.x[i : i + 30]
        y = self.y[i + 30]

        return x, y

dataset = Custom_Dataset(train_csv)

# print(dataset)
# print(type(dataset))

# print(dataset[0])
# print(dataset[0][0].shape) # (30, 3)
# print(dataset[0][1]) # 94
# print(len(dataset)) # 967
# print(dataset[937])

# train_loader = DataLoader(dataset, batch_size = 32)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(input_size = 3,
                          hidden_size = 64,
                          num_layers = 5,
                          batch_first = True)

        self.fc1 = nn.Linear(in_features = 30 * 64, out_features = 32)
        self.fc2 = nn.Linear(in_features = 32, out_features = 1)

        self.relu = nn.ReLU()

    def forward(self, x, h0):
        x, hn = self.rnn(x, h0)

        x = torch.reshape(x, (x.shape[0], -1))

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

model = RNN().to(DEVICE)

# train
# optimizer = optim.Adam(params = model.parameters(), lr = 0.001)

# for epoch in range(1, 201):
#     iterator = tqdm.tqdm(train_loader)

#     for x, y in iterator:
#         optimizer.zero_grad()
        
#         h0 = torch.zeros(5, x.shape[0], 64).to(DEVICE) # num_layers, batch_size, hidden_size = 5, 32, 64

#         hypothesis = model(x.type(torch.FloatTensor).to(DEVICE), h0)

#         loss = nn.MSELoss()(hypothesis, y.type(torch.FloatTensor).to(DEVICE))

#         loss.backward()

#         optimizer.step()

#         iterator.set_description(f'epoch:{epoch:0.10f} loss:{loss.item():0.10f}')

PATH_SAVE = './_save/torch/'

# torch.save(model.state_dict(), PATH_SAVE + 't22.pth')

#4 평가 예측
train_loader = DataLoader(dataset, batch_size = 1)

y_predict = []
y_test_list = []

total_loss = 0
r2 = 0

with torch.no_grad():
    model.load_state_dict(torch.load(PATH_SAVE + 't22.pth', map_location = DEVICE, weights_only = True))

    for x_test, y_test in train_loader:
        h0 = torch.zeros(5, x_test.shape[0], 64).to(DEVICE)

        y_pred = model(x_test.type(torch.FloatTensor).to(DEVICE), h0)

        y_predict.append((y_pred.item()))

        loss = nn.MSELoss()(y_pred, y_test.type(torch.FloatTensor).to(DEVICE))

        total_loss += loss

        y_test_list.append(y_test)

print('total_loss : ', total_loss.item() / len(train_loader))

print("rs :", r2_score(y_test_list, y_predict))