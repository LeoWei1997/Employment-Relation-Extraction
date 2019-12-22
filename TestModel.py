import torch
import re
import BiLSTM_ATT
from torch.autograd import Variable
import numpy as np
# checkpoint=torch.load("./model/model_01.pkl")
# model = BiLSTM_ATT()  # 传入结构配置和词典
# model.load_state_dict()

# torch.manual_seed(1)
model=torch.load("./model/model_过拟合.pt")
model.eval()

while True:
    id_sent=input('sent: ')
    id_sent=id_sent.split(' ')
    id_sent.pop()
    id_sent=[int(i) for i in id_sent]
    id_sent=Variable(torch.tensor(id_sent).repeat(128,1))  # 我是一条一条输入，只好扩充成一批
    print('sent len: ',len(id_sent))
    pos1=input('pos1: ')
    pos1=pos1.split(' ')
    pos1.pop()
    pos1=[int(i) for i in pos1]
    pos1=Variable(torch.tensor(pos1).repeat(128,1))
    print('pos1 len: ',len(pos1))
    pos2 = input('pos2: ')
    pos2 = pos2.split(' ')
    pos2.pop()
    pos2=[int(i) for i in pos2]
    pos2=Variable(torch.tensor(pos2).repeat(128,1))
    print('pos1 len: ', len(pos2))
    with torch.no_grad():
        y = model.forward(id_sent, pos1, pos2)
        y = np.argmax(y.data.numpy(), axis=1)
    print('预测: ',y)

