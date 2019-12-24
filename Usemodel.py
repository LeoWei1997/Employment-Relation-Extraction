import os
import re

import numpy as np
import torch
from pyltp import NamedEntityRecognizer
from pyltp import Postagger
from pyltp import Segmentor
from torch.autograd import Variable

LTP_DIR = 'C:/Users/Leqsott/PycharmProjects/Employment-Relation-Extraction/ltp_data_v3.4.0'
cws_model_path = os.path.join(LTP_DIR, 'cws.model')  # 分词模型路径
pos_model_path = os.path.join(LTP_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
ner_model_path = os.path.join(LTP_DIR, 'ner.model')  # 词性标注模型路径，模型名称为`ner.model`

THRESHOLD = 0.5

segmentor = Segmentor()  # 初始化
segmentor.load_with_lexicon(cws_model_path, 'NE.txt')  # 辅助分词

postagger = Postagger()  # 初始化实例
postagger.load(pos_model_path)  # 加载模型

recognizer = NamedEntityRecognizer()
recognizer.load(ner_model_path)

kickout = re.compile('([０-９]*日电)|([０-９]*[年日月时分])|[０-９0-9，、。（）〈〉‘’“”：∶；！?？％《》『』{}●/／nrtsz\n×’．…了]')

trainword = []
news = open('./datatonet/pre_sent50.txt', 'r', encoding='utf-8').readlines()
for i in range(0, len(news)):
    news[i] = news[i].split(' ')  # 最后一个元是空字符
    news[i].pop()
    for w in news[i]:
        if w not in trainword:
            trainword.append(w)  # 训练集词集合


def cutsent(sent, s, e):
    # 初始字段是nr-nt,扩充至len 50
    ns = s
    ne = e
    while (ne - ns) != 49:
        if ns != 0:  # 优先向左扩充
            ns -= 1
        elif ne != len(sent):
            ne += 1
    return sent[ns:ne + 1]


def expandsent(sent):
    while len(sent) != 50:
        sent.append('奥力给')
    return sent


def getpos(sent50, nrt):
    try:
        nrtidx = sent50.index(nrt)
    except:
        print('can not getpos: ', sent50)  # 如果报错去修改LabeledData.txt!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        print(nrt)
    pos = [i - nrtidx + 49 for i in range(50)]
    return pos


# 清洗
def cleanSent(sent):
    sent = re.sub(kickout, '', sent)
    return sent


# 分词
def segSent(sent):
    segSent = list(segmentor.segment(sent))  # ['国家主席', '江泽民', '访问', '了', '美国']
    return segSent


# 词性标注
def posSent(segsent):
    posSent = list(postagger.postag(segsent))
    return posSent


# 实体识别
def nerSent(segSent, posSent):
    nerSent = list(recognizer.recognize(segSent, posSent))  # 实体标注
    return nerSent


# 文本序号化,序号向量化交给神经网络来做
def getSentEncodeIdx(sent):
    sentencode = []
    for w in sent:
        try:
            wencode = trainword.index(w)
        except:
            wencode = trainword.index('奥力给')
        sentencode.append(int(wencode))  # 从词集找序号
    # sentencode = np.asarray(sentencode)
    return sentencode


# model=torch.load('./model/model_epoch20.pkl')
model = torch.load('./model/model_过拟合.pt')
model.eval()
while True:
    ni = []  # 组织
    nh = []  # 人名
    sent = input('输入语句：')
    sent = cleanSent(sent)
    print(sent)
    segsent = segSent(sent)
    print(segsent)
    possent = posSent(segsent)
    print(possent)
    nersent = nerSent(segsent, possent)
    print(nersent)
    # 获取ni，nh
    tmpforlongni = ''
    tmpforlongnh = ''
    for idx in range(len(nersent)):
        s = nersent[idx]
        if s == 'S-Ni':
            ni.append(segsent[idx])
        elif s == 'S-Nh':
            nh.append(segsent[idx])
        elif s == 'B-Ni' or s == 'I-Ni':
            tmpforlongni += segsent[idx]
        elif s == 'E-Ni':
            tmpforlongni += segsent[idx]
            ni.append(tmpforlongni)
            tmpforlongni = ''
        elif s == 'B-Nh' or s == 'I-Nh':
            tmpforlongnh += segsent[idx]
        elif s == 'E-Nh':
            tmpforlongnh += segsent[idx]
            nh.append(tmpforlongnh)
            tmpforlongnh = ''
    space_sent = ' '.join(segsent)
    print('now link togther for reunion: ', space_sent)
    for h in nh:
        print(h)
        nhrule = re.compile('( )?'.join(h))  # '江( )?泽( )?民'
        space_sent = re.sub(nhrule, h, space_sent)
    for i in ni:
        print(i)
        nirule = re.compile('( )?'.join(i))  # '中( )?华( )?全( )?国( )?总( )?工( )?会'
        space_sent = re.sub(nirule, i, space_sent)
    space_sent = cleanSent(space_sent).split(' ')
    space_sent = [i for i in space_sent if i]  # 剔除空字符
    # print('sent after clean: ', space_sent)

    answersheet = [[0, -1] for x in range(0, 100)]

    # 判断关系
    for h in nh:
        for i in ni:
            if len(space_sent) > 50:  # 需要裁剪
                try:
                    hidx = space_sent.index(h)
                    iidx = space_sent.index(i)
                except:
                    print('没找到实体')
                    break
                if iidx < hidx:
                    sent50 = cutsent(space_sent, iidx, hidx)
                    hpos = getpos(sent50, h)  # 人位置
                    ipos = getpos(sent50, i)  # 机构位置
                else:
                    sent50 = cutsent(space_sent, hidx, iidx)
                    hpos = getpos(sent50, h)  # 人位置
                    ipos = getpos(sent50, i)  # 机构位置
            else:  # 需要扩充
                sent50 = expandsent(space_sent)
                hpos = getpos(sent50, h)  # 人位置
                ipos = getpos(sent50, i)  # 机构位置

            sentencode = getSentEncodeIdx(sent50)
            batch_sentencode = Variable(torch.tensor(sentencode).repeat(128, 1))  # 我是一条一条输入，只好扩充成一批
            # print('sent len: ', batch_sentencode.shape)
            batch_hpos = Variable(torch.tensor(hpos).repeat(128, 1))
            batch_ipos = Variable(torch.tensor(ipos).repeat(128, 1))
            # print('pos len:', batch_hpos.shape)
            with torch.no_grad():
                y = model.forward(batch_sentencode, batch_hpos, batch_ipos)
                y = np.argmax(y.data.numpy(), axis=1)
            yi = 0
            total = 0
            for n in y:
                if n == 1:
                    yi += 1
                total += 1
            valrate = yi / total
            # print(h, ' and ', i, '预测概率 1：0  ', valrate, 1 - valrate)
            print(h, ' 受雇于 ', i, '预测概率 ：', valrate)
            peopos = 50 - hpos[0] - 1
            orgpos = 50 - ipos[0] - 1
            # print("人位置：", peopos)
            # print("机构位置：", orgpos)
            if answersheet[peopos][1] == -1 or answersheet[peopos][0] < valrate:  # 人名还未匹配到机构 或 答案几率 < 当前几率
                answersheet[peopos][0] = valrate
                answersheet[peopos][1] = orgpos
            elif answersheet[peopos][0] == valrate:  # 答案几率 == 当前几率
                if orgpos < peopos < answersheet[peopos][1] \
                        or answersheet[peopos][1] < orgpos < peopos \
                        or peopos < orgpos < answersheet[peopos][1]:
                    answersheet[peopos][1] = orgpos

    for i in range(0, 100):
        if answersheet[i][1] != -1 and answersheet[i][0] > THRESHOLD:
            print(sent50[answersheet[i][1]], "雇佣了", sent50[i])

# while True:
#     id_sent=input('sent: ')
#     id_sent=id_sent.split(' ')
#     id_sent.pop()
#     id_sent=[int(i) for i in id_sent]
#     id_sent=Variable(torch.tensor(id_sent).repeat(128,1))
#     print('sent len: ',len(id_sent))
#     pos1=input('pos1: ')
#     pos1=pos1.split(' ')
#     pos1.pop()
#     pos1=[int(i) for i in pos1]
#     pos1=Variable(torch.tensor(pos1).repeat(128,1))
#     print('pos1 len: ',len(pos1))
#     pos2 = input('pos2: ')
#     pos2 = pos2.split(' ')
#     pos2.pop()
#     pos2=[int(i) for i in pos2]
#     pos2=Variable(torch.tensor(pos2).repeat(128,1))
#     print('pos1 len: ', len(pos2))
#     with torch.no_grad():
#         y = model.forward(id_sent, pos1, pos2)
#         y = np.argmax(y.data.numpy(), axis=1)
#     print('预测: ',y)
