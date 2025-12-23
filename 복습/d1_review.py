import numpy as np
import torch

# x = 3
x = torch.tensor(3.0, requires_grad=True)
w = torch.tensor(1.0, requires_grad=True)
z = w * x
out = z ** 2
out.backward()
# print('out을 w로 미분한 값', w.grad)
# print('out을 x로 미분한 값', x.grad)

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# print(x.size()) # torch.Size([3, 3])
# print(x.ndimension()) # 2
# print(x.shape) # torch.Size([3, 3])

x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
x = torch.unsqueeze(x, 1) # unsqueeze: 기존 데이터는 그대로 두고, 크기 1인 차원 하나 추가. 2번째 argument는 dim을 의미
# print(x.shape) # torch.Size([3, 1, 3])

x = torch.squeeze(x)
# print(x.shape) # torch.Size([3, 3])
x = torch.unsqueeze(x, 2)
# print(x.shape) # torch.Size([3, 3, 1])

x = torch.squeeze(x)
x = torch.unsqueeze(x, 0)
# print(x.shape) # torch.Size([1, 3, 3])

x = torch.squeeze(x)
# print(x + np.array([1, 1, 1])) # cpu에서는 되지만 gpu에선 에러. 그리고 그래프에서 numpy는 추적 안됨!

# print(x + torch.tensor([1,1,1], device=x.device, dtype=x.dtype))
# print(x + torch.from_numpy([1, 1, 1]).to(x)) # .to(): 텐서의 device 와 dtype 을 다른 텐서와 맞추기 위한 연산

# print(x.view(9, 1, 1))

# print(x.reshape(1, 9, 1))

CUDA = torch.cuda.is_available()
# print("cuda:", CUDA)

w = torch.randn(5, 3, dtype=torch.float)
x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# print(w.shape, x.shape) # torch.Size([5, 3]) torch.Size([3, 2])

b = torch.randn(5, 2, dtype=torch.float)
# print(b.shape) # torch.Size([5, 2])

wx = torch.mm(w, x)
z = wx + b
# print(z)

from sklearn.datasets import make_blobs
n_dim = 2
tr_x, tr_y = make_blobs(n_samples=80, n_features=2, centers=[[1, 1], [-1, -1], [1, -1], [-1, 1]], shuffle=True, cluster_std=0.3)
tt_x, tt_y = make_blobs(n_samples=80, n_features=2, centers=[[1, 1], [-1, -1], [1, -1], [-1, 1]], shuffle=True, cluster_std=0.3)

# print(tr_x, tr_y)

def label_map(y, from_, n):
    ck_y = y.copy()
    for i in from_:
        ck_y[y==i] = n
    return ck_y

tr_y = label_map(tr_y, [0, 1], 0) # 0과 1은 0으로 라벨링
tr_y = label_map(tr_y, [2, 3], 1) # 2와 3은 1로 라벨링
tt_y = label_map(tt_y, [0, 1], 0) # 0과 1은 0으로 라벨링
tt_y = label_map(tt_y, [2, 3], 1) # 2와 3은 1로 라벨링

# print(tr_x.shape, tr_y.shape) # (80, 2) (80,)

# print(np.unique(tr_y))

import matplotlib.pyplot as plt
def v_data(x,y=None,c='r'):
    if y is None:
        y=[None]*len(x)# [None for _ in range(len(x))]
    for ck_x,ck_y in zip(x,y):
        if ck_y is None:
            plt.plot(ck_x[0],ck_x[1],'*',markerfacecolor='none',markeredgecolor=c)
        else:
            plt.plot(ck_x[0],ck_x[1], 'o'+c if ck_y==0 else '+'+c )

# v_data(tr_x,tr_y)
# plt.show()

t_tr_x = torch.FloatTensor(tr_x)
t_tt_x = torch.FloatTensor(tt_x)
t_tr_y = torch.FloatTensor(tr_y)
t_tt_y = torch.FloatTensor(tt_y)
# print(t_tr_x.shape, t_tr_y.shape, t_tt_x.shape, t_tt_y.shape) # torch.Size([80, 2]) torch.Size([80]) torch.Size([80, 2]) torch.Size([80])

class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.l1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, input_tensor):
        l1 = self.l1(input_tensor)
        relu = self.relu(l1)
        l2 = self.l2(relu)
        output = self.sigmoid(l2)
        return output
    
model = NeuralNet(2, 5)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
epochs = 1000

ty = t_tt_y
# print(ty) # 0과 1
py = model(t_tt_x)
# print(py) # 확률값
# print(ty.shape, py.shape) # torch.Size([80]) torch.Size([80, 1])
# L(p,t)=−[tlog(p)+(1−t)log(1−p)]
# print(criterion(ty, py.squeeze()).item()) # 49.46245193481445. log(0) → log(1e-7) 같은 값으로 처리

# -np.log(0.1) 만 해도 2.3 정도고 인자에 0.00000001 대입하면 18.420680743952367임
# log값이 49라는 건 0에 가까운 수가 들어간 것!
# print(-np.log(0.3)) # 1.2039728043259361
# print(-np.log(0.1)) # 2.3025850929940455
# print(np.exp(1)) # 2.718281828459045


test_loss = criterion(py.squeeze(), ty)
# print(test_loss.item()) # 0.6931719183921814

'''
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    tr_output = model(t_tr_x)
    tr_loss = criterion(t_tr_y, tr_output.squeeze())
    if epoch % 100 == 0:
        print(f"epoch: {epoch} loss: {tr_loss.item()}")
    tr_loss.backward()
    optimizer.step()
'''


class NeuralNet2(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        h = self.relu(self.l1(x))
        logit = self.l2(h)          # sigmoid 전 값
        prob = self.sigmoid(logit) # sigmoid 후 값
        return logit, prob

model = NeuralNet2(2, 5)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)

'''
for epoch in range(5):  # 일부 epoch만
    model.train()
    optimizer.zero_grad()

    logit, prob = model(t_tr_x)

    # 디버깅 출력
    print(f"\n[epoch {epoch}]")
    print("logit min/max:", logit.min().item(), logit.max().item())
    print("prob  min/max:", prob.min().item(), prob.max().item())

    # 일부러 잘못된 순서 (문제 재현)
    loss = criterion(t_tr_y, prob.squeeze())
    print("loss:", loss.item())

    loss.backward()
    optimizer.step()
'''