# coding:utf8
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)


class BiLSTM_ATT(nn.Module):
    def __init__(self, config, embedding_pre):
        super(BiLSTM_ATT, self).__init__()
        self.batch = config['BATCH']

        self.embedding_size = config['EMBEDDING_SIZE']
        self.embedding_dim = config['EMBEDDING_DIM']

        self.hidden_dim = config['HIDDEN_DIM']
        self.tag_size = config['TAG_SIZE']

        self.pos_size = config['POS_SIZE']
        self.pos_dim = config['POS_DIM']

        self.pretrained = config['pretrained']
        if self.pretrained:
            # 词嵌入词典,使用外部预训练词典
            self.word_embeds = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_pre), freeze=False)
        else:
            # 自定的词嵌入没经过大量学习,没什么意义
            self.word_embeds = nn.Embedding(self.embedding_size, self.embedding_dim)

        self.pos1_embeds = nn.Embedding(self.pos_size, self.pos_dim)
        self.pos2_embeds = nn.Embedding(self.pos_size, self.pos_dim)
        self.relation_embeds = nn.Embedding(self.tag_size, self.hidden_dim)  # Bi隐层拼接成200维,所以这里每种关系都编码成200维

        # 每个词输入维度:词向量维度+2*位置维度, 隐层维度: 100, 隐层数: 1, 双向
        # 注意RNN中隐层是横向依次传播的,传到末尾再传到下一层
        self.lstm = nn.LSTM(input_size=self.embedding_dim + self.pos_dim * 2, hidden_size=self.hidden_dim // 2,
                            num_layers=1, bidirectional=True,dropout=0)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)  # 200维隐层输出映射到10种关系的函数

        # 防止过拟合,每个神经元0.5概率休眠
        # self.dropout_emb = nn.Dropout(p=0.5)
        # self.dropout_lstm = nn.Dropout(p=0.5)
        # self.dropout_att = nn.Dropout(p=0.5)

        # 这是在搞什么玩意
        self.hidden = self.init_hidden()

        # att层是怎么加进来的??
        # 隐层到att层的权值矩阵:Bi隐层拼接出1*200,
        self.att_weight = nn.Parameter(torch.randn(self.batch, 1, self.hidden_dim))
        self.relation_bias = nn.Parameter(torch.randn(self.batch, self.tag_size, 1))

    def init_hidden(self):
        return torch.randn(2, self.batch, self.hidden_dim // 2)

    def init_hidden_lstm(self):  # 为什么是两个 ????????????????????????????????????????????
        # 其实是横向起始神经元的h,c, 2是因为双向两个
        # 横向传播
        # return (torch.randn(2, self.batch, self.hidden_dim // 2),
        #         torch.randn(2, self.batch, self.hidden_dim // 2))
        return (torch.zeros(2, self.batch, self.hidden_dim // 2),
                torch.zeros(2, self.batch, self.hidden_dim // 2))

    # lstm输出到att层
    def attention(self, H):
        M = F.tanh(H)
        # batch*1*200 X batch*100*len 这两个怎么乘?????????????? 是不是Bi所以100拼接成200??????????
        # 沿着第三维度计算softmax
        a = F.softmax(torch.bmm(self.att_weight, M), 2)  # 一般是input*weight,这里反了?????? 反了则weight无法控制下一层神经元个数呀
        # a的大小是: batch*1*len ATT层
        a = torch.transpose(a, 1, 2)  # batch*len*1
        # 返回 batch*200*1
        return torch.bmm(H, a)

    def forward(self, sentence, pos1, pos2):
        # print('hello')

        self.hidden = self.init_hidden_lstm()  # 每次批训练都初始化隐层????????????????? 应该是初始化起点h,c,权值并没有初始化

        # 这里sentence和pos有三个维度!!!! 句数*一句话词数*词向量维度, batch*len*dim !!
        # 注意batch在第二维度有利于内存计算,是RNN默认的,这里的输入需要调整适应
        # print(self.pos1_embeds(pos1).shape)
        embeds = torch.cat((self.word_embeds(sentence), self.pos1_embeds(pos1), self.pos2_embeds(pos2)),2)  # 2表示在第三维度上链接词向量和位置向量len*(vec_dim+2*pos_dim)

        embeds = torch.transpose(embeds, 0, 1)  # 转置成为len*batch*dim符合RNN输入要求

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)  # self.hidden=(h,c)
        # out 是所有横向元,hidden是末尾两个元
        # len*batch*dim-> len*batch*100
        # 转置,batch*len*100,
        # 转置,batch*100*len
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)

        # lstm_out = self.dropout_lstm(lstm_out)

        att_out = F.tanh(self.attention(lstm_out))  # batch*200*1

        # att_out = self.dropout_att(att_out)

        relation = torch.tensor([i for i in range(self.tag_size)], dtype=torch.long).repeat(self.batch,1)  # 重复成 batch*tag_size

        relation = self.relation_embeds(relation)  # batch * tag_size *200

        res = torch.add(torch.bmm(relation, att_out), self.relation_bias)  # batch*tag_size*1+bias

        res = F.softmax(res, 1)  # 沿着tag_size计算softmax(每种tag概率), batch*tag_size*1

        return res.view(self.batch, -1)  # 展成batch行,列数根据数据数目定,即 batch*tag_size
