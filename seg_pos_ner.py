# -*- coding: utf-8 -*-
# 读取cleanNews.txt, 分词,词性标注,实体识别
# 这里分词十分垃圾,分完之后对于目标词还是要进行重组

news=open('./cleanNews.txt','r',encoding='utf-8').readlines()

segednews=open('./segedNews.txt','w',encoding='utf-8')

# 辅助分词很关键,千万别把 沙曼维亚济 分成 沙曼 维亚济
import os
LTP_DIR = 'D:\Python-dev\ltp_data_v3.4.0'
cws_model_path = os.path.join(LTP_DIR, 'cws.model') # 分词模型路径

from pyltp import Segmentor
segmentor = Segmentor() # 初始化
segmentor.load_with_lexicon(cws_model_path,'./NE.txt') # 辅助分词
for idx,sent in enumerate(news):
    segSent = list(segmentor.segment(sent)) # ['国家主席', '江泽民', '访问', '了', '美国']
    news[idx]=segSent # ['国家主席江泽民访问了美国']->[['国家主席', '江泽民', '访问', '了', '美国']]

for n in news:
    for word in n:
        segednews.write(word+' ')
    segednews.write('\n')
segednews.close()


# 这个垃圾分词,用了外部词典还是把不该分的分了!!!!
import re
news=open('./segedNews.txt','r',encoding='utf-8').readlines()
# 为 企业 改革 发展 建功立业 本报 北京 讯 中华 全国 总工会 今 发出 致 全国 各族 职工 慰问信 更加 紧密 地 团结 在 以 江泽民 同志 为 核心 的 党中央 周围
# 中华全国总工会被分词了,玩个屁
# 我知道了,用正则? none or once ' '就可以抢救回来!!!!
relations=open('./relation_pos_neg.txt','r',encoding='utf-8').readlines()
realSegedNews=open('./realSegedNews.txt','w',encoding='utf-8')
for r in relations:
    r = r.split(' ')  # ['0', '江泽民', '中华全国总工会', '0', '']
    newid = int(r[0])
    nr = r[1]
    nt = r[2]
    # res = int(r[3])

    nrrule = re.compile('( )?'.join(nr))  # '江( )?泽( )?民'
    ntrule = re.compile('( )?'.join(nt))  # '中( )?华( )?全( )?国( )?总( )?工( )?会'
    news[newid] = re.sub(nrrule, nr, news[newid])
    # if nt=='中国美术家协会':  # 发现没有替换, 原因见 cleanTest.py line 116
    #     print('wocao!!!')
    #     print('( )?'.join(nt))
    #     print(news[newid])
    #     print(re.search(ntrule,news[newid]))
    news[newid] = re.sub(ntrule, nt, news[newid])



for n in news:
    realSegedNews.write(n)
realSegedNews.close()
# 多么完美


