# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim

import torch.utils.data as D
from torch.autograd import Variable
from BiLSTM_ATT import BiLSTM_ATT

EMB_DIM=100  # 词向量维度
POS_SIZE=99  # 原文是82,为什么?pos_size用于自动嵌入,不应该是-49 到 +49吗,共99,补充:必须全部正值化,0-98
POS_DIM=25   # 用于自动嵌入,99种位置码,嵌成10维,好像太多----------------------------------------------------

HID_DIM= 200 # Bi,所以是两个100拼接

TAG_SIZE=2

BATCH =128  # 单次输入批数----------------------------------------------------------------
EPOCHS= 100  # 轮数----------------------------------------------------------
config={}
config['EMBEDDING_SIZE'] = 'no_use'
config['EMBEDDING_DIM'] = EMB_DIM
config['POS_SIZE'] = POS_SIZE
config['POS_DIM'] = POS_DIM
config['HIDDEN_DIM'] = HID_DIM
config['TAG_SIZE'] = TAG_SIZE
config['BATCH'] = BATCH
config["pretrained"]= True  # 使用外部词向量

learning_rate =0.0005#0.0003

embedding_pre=[]
# 组装句向量(len=50)准备送到神经网络
trainword=[]  # vec.txt太大了,不该全传给神经网络,训练集中用到多少就传多少
#
# relation2id={'非雇员':0,'雇员':1}
# sentlen=50
# count=[0,0]  # 计数划分训练集测试集
vec_unknow=[1 for i in range(100)]  # 没找到的词,向量化为全1
# news=open('./datatonet/finalsent50.txt','r',encoding='utf-8').readlines()
# poses=open('./datatonet/finalpos.txt','r',encoding='utf-8').readlines()
# relas=open('./datatonet/finalrela.txt','r',encoding='utf-8').readlines()
news=open('./datatonet/pre_sent50.txt','r',encoding='utf-8').readlines()
poses=open('./datatonet/pre_pos.txt','r',encoding='utf-8').readlines()
relas=open('./datatonet/pre_rela.txt','r',encoding='utf-8').readlines()
# 词语集合化
for i in range(0,len(news)):
    news[i]=news[i].split(' ')  # 最后一个元是空字符
    news[i].pop()
    for w in news[i]:
        if w not in trainword:
            trainword.append(w)  # 训练集词集合

# 加载词向量模型,把trainword转成词向量
from gensim.models import KeyedVectors
import numpy as np
# 用新版加载模型
w2v_model=KeyedVectors.load('word2vec_model')  # 词向量模型
# 很多词语都不在word2vec中
for w in trainword:
    try:
        vec=list(w2v_model[w])  # 从词向量模型中找词
        embedding_pre.append(vec)
    except:
        embedding_pre.append(vec_unknow)
embedding_pre=np.asarray(embedding_pre)  # 为什么转成np序列??????????????????????????
print('embedding_pre shape: ',embedding_pre.shape)

# 文本序号化,序号向量化交给神经网络来做
train=[]
for n in news:
    nid=[]
    for w in n:
        nid.append(trainword.index(w))  # 不是从word2vec中找序号,而是从词集找序号
    train.append(nid)
# pre_sent50id=open()-------------------------------------------------------------------------------
train = np.asarray(train)  # 同样的,转成np序列???????????????????????????????

# 位置分割成pos1,2
pos1=[]
pos2=[]
for i in range(0,len(poses)):
    poses[i]=poses[i].split(' ')  # 最后一个元是空字符
    poses[i].pop()
    poses[i]=[int(i) for i in poses[i]]  # 字符转整数
    if i&1:  # 奇数,pos2
        pos2.append(poses[i])
    else:  # 偶数,pos1
        pos1.append(poses[i])
pos1 = np.asarray(pos1)
pos2 = np.asarray(pos2)

# 关系
relas=[int(i) for i in relas]
relas=np.asarray(relas)

# 构建网络
model = BiLSTM_ATT(config,embedding_pre)  # 传入结构配置和词典
optimizer =optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-5)  # 设置网络优化参数
criterion = nn.CrossEntropyLoss(reduction='mean')  # Loss设置,什么意思??????????????????????

# 150 对关系用于测试,其实更好的办法是根据rela的计数另外写入test.txt,这里简化,直接根据序号取末尾的
test_at_least_num=1500  # 具体并不是150,因为batch会抹掉零头加入,然后再抹掉零头
# 组装数据
keep=(len(train)-test_at_least_num)-(len(train)-test_at_least_num)%BATCH
# 加载之后train貌似就没了,还想着捡剩饭[keep:]是不行了,必须拷贝一份
t_train=train[keep:]
t_pos1=pos1[keep:]
t_pos2=pos2[keep:]
t_relas=relas[keep:]

train=torch.LongTensor(train[:keep])  # 抹掉零头!?? train是句子的词id形式,长度一直是50,共有1047条,分批32
pos1=torch.LongTensor(pos1[:keep])
pos2=torch.LongTensor(pos2[:keep])
relas=torch.LongTensor(relas[:keep])
train_datasets= D.TensorDataset(train,pos1,pos2,relas)  # 组装
train_dataloader = D.DataLoader(train_datasets,BATCH,True)  # 加载,,num_workers=2,不要多线程,会出错


# 测试集,从keep到len(train)
# test_total_num = len(train) - keep
# test_total_num = test_total_num - test_total_num%BATCH  # 再次抹掉零头,还是有剩余,太浪费
t_keep=len(t_train)-len(t_train)%BATCH
test = torch.LongTensor(t_train[:t_keep])
t_pos1 = torch.LongTensor(t_pos1[:t_keep])
t_pos2 = torch.LongTensor(t_pos2[:t_keep])
t_relas = torch.LongTensor(t_relas[:t_keep])
test_datasets = D.TensorDataset(test,t_pos1,t_pos2,t_relas)
test_dataloader = D.DataLoader(test_datasets,BATCH,True)  # ,num_workers=2

import re
for epoch in range(EPOCHS):

    # inputinfo=open('./datatonet/inputinfo.txt','w',encoding='utf-8')

    print('轮数: ',epoch)
    acc=0
    total=0


    for sent,p1,p2,rela in train_dataloader:
        # print('看清楚',sent.shape)
        # inputinfo.write(str(total)+'\n')
        # for wid in sent[0]:
        #     inputinfo.write(trainword[wid]+' ')
        # inputinfo.write('\n')
        # for wid in sent[0]:
        #     wid=re.sub('[tensor()]','',str(wid))
        #     inputinfo.write(wid+' ')
        # inputinfo.write('\n')
        # for p in p1[0]:
        #     p = re.sub('[tensor()]', '', str(p))
        #     inputinfo.write(p+' ')
        # inputinfo.write('\n')
        # for p in p2[0]:
        #     p = re.sub('[tensor()]', '', str(p))
        #     inputinfo.write(p+' ')
        # inputinfo.write('\n')
        # inputinfo.write(re.sub('[tensor()]','',str(rela[0])))
        # inputinfo.write('\n')

        model.train()  # train模式？？--------------------------

        sent=Variable(sent)  # 为什么要variable???????????????????
        p1=Variable(p1)
        p2=Variable(p2)
        rela=Variable(rela)

        y=model(sent,p1,p2)  # 其实是调用了forward函数,返回batch*tag_size,32组2种类别概率
        loss=criterion(y,rela)  # 损失函数,y:batch*2,rela: batch*1,这怎么算??????????
        optimizer.zero_grad()  # 清除梯度？？？？？？？？？？？？？？？
        loss.backward()
        # 其实,loss(x,class)=-log(exp(x[class])/sum(exp(x[i])))= -x[class]+log(sum(exp(x[i]))
        # class是x 总tag_size维度中的一支
        optimizer.step()

        y=np.argmax(y.data.numpy(),axis=1)  # ?????????????? 取最大tag的位置吗,应该是

        for y1,y2 in zip(y,rela):
            if y1==y2:  # 不同维度能比较大小,我服了
                # print('猜对了%d==%d'%(y1,y2))  # 把猜对是1的提出来看看能不能测试对
                # if y1==1:
                #     print('当前： ',total)
                acc+=1
            total+=1
    print('当前轮训练准确度 ',100*float(acc)/total,"%")

    model.eval()

    acc_t = 0
    total_t =0
    cnt_predict=[0,0]  # 用于计算,precision,recall
    cnt_total=[0,0]
    cnt_right=[0,0]
    for sent,p1,p2,rela in test_dataloader:
        sent =Variable(sent)
        p1 = Variable(p1)
        p2 = Variable(p2)
        rela = Variable(rela)

        y =model(sent,p1,p2)
        y=np.argmax(y.data.numpy(),axis=1)

        for y1,y2 in zip(y,rela):
            cnt_predict[y1]+=1
            cnt_total[y2]+=1
            if y1==y2:
                cnt_right[y1]+=1

    precision=[0,0]
    recall=[0,0]
    for i in range(len(cnt_predict)):
        if cnt_predict[i]!=0:
            precision[i]=float(cnt_right[i])/cnt_predict[i]
        if cnt_total[i]!=0:
            recall[i]=float(cnt_right[i])/cnt_total[i]

    precision=sum(precision)/TAG_SIZE  # 种类,平均准确率
    recall=sum(recall)/TAG_SIZE
    print('准确率: ',precision)
    print('召回率: ',recall)
    print('f: ',(2*precision*recall)/(precision+recall))

    if epoch % 20 == 0:
        model_name = "./model/model_epoch" + str(epoch) + ".pkl"
        torch.save(model, model_name)
        print(model_name, "has been saved")

    # inputinfo.close()

torch.save(model, "./model/model_01.pt")
print("model state_dict has been saved")

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

