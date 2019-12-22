# -*- coding: utf-8 -*-


# 0 江泽民 中华全国总工会 0
relations=open('./relation_pos_neg.txt','r',encoding='utf-8').readlines()
# 为 企业 改革 发展 建功立业 本报 北京 讯 中华全国总工会 今 发出 致 全国 各族 职工 慰问信 更加 紧密 地 团结 在 以 江泽民 同志 为 核心 的 党中央 周围
news=open('./realSegedNews.txt','r',encoding='utf-8').readlines()

for i in range(0,len(news)):
    news[i]=news[i].split(' ')  # 最后一个元是空字符
    news[i].pop()


sentforrela=[]
rela=[]
nrtpos=[]

def cutsent(sent,s,e):
    # 初始字段是nr-nt,扩充至len 50
    ns=s
    ne=e
    while  (ne-ns) != 49:
        if ns!=0:  # 优先向左扩充
            ns-=1
        elif ne!=len(sent):
            ne+=1
    return sent[ns:ne+1]

def expandsent(sent):
    while len(sent)!=50:
        sent.append('奥力给')
    return sent

def getpos(sent50,nrt):
    try:
        nrtidx=sent50.index(nrt)
    except:
        print('getpos: ',sent50)  # 如果报错去修改LabeledData.txt!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        print(nrt)
    pos=[i-nrtidx for i in range(50)]
    return pos

import random
random.seed(123)
for r in relations:
    r=r.split(' ')  # ['0', '江泽民', '中华全国总工会', '0', '']
    newid=int(r[0])
    nr=r[1]
    nt=r[2]
    res=int(r[3])
    if res==0:
        if random.random()<0.766:
            continue

    if len(news[newid])>50:  # 需要裁剪
        try:
            nridx=news[newid].index(nr)
            ntidx=news[newid].index(nt)
        except:
            print('没找到: ',r)  # 如果报错去修改LabeledData.txt!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            break
        if abs(ntidx-nridx)>15:  # 两个实体相距20词,直接不要这对关系了. 反例太多,可能需要改一下!!!!!!!!!!!!!!!!!!!!!----------------------------
            continue
        elif ntidx<nridx:
            sent50=cutsent(news[newid],ntidx,nridx)
            sentforrela.append(sent50)
            rela.append([nr,nt,res])
            nrtpos.append(getpos(sent50,nr))
            nrtpos.append(getpos(sent50,nt))
        else:
            sent50 = cutsent(news[newid], nridx, ntidx)
            sentforrela.append(sent50)
            rela.append([nr, nt, res])
            nrtpos.append(getpos(sent50, nr))
            nrtpos.append(getpos(sent50, nt))
    else:  # 需要补充
        sent50=expandsent(news[newid])
        sentforrela.append(sent50)
        rela.append([nr, nt, res])
        nrtpos.append(getpos(sent50, nr))
        nrtpos.append(getpos(sent50, nt))

# finalsent50=open('./datatonet/finalsent50.txt','w',encoding='utf-8')
# finalrela=open('./datatonet/finalrela.txt','w',encoding='utf-8')
# finalpos=open('./datatonet/finalpos.txt','w',encoding='utf-8')
finalsent50=open('./datatonet/pre_sent50.txt','w',encoding='utf-8')
finalrela=open('./datatonet/pre_rela.txt','w',encoding='utf-8')
finalpos=open('./datatonet/pre_pos.txt','w',encoding='utf-8')
for sent50 in sentforrela:
    for w in sent50:
        finalsent50.write(w+' ')
    finalsent50.write('\n')
finalsent50.close()
ling=0
yi=0
for rtr in rela:
    # finalrela.write(str(rtr[2])+' '+rtr[0]+' '+rtr[1])
    if rtr[2]==1:
        yi+=1
    else:
        ling+=1
    finalrela.write(str(rtr[2]))
    finalrela.write('\n')
finalrela.close()
print('1: ',yi,' 0: ',ling)
for p1or2 in nrtpos:
    for x in p1or2:
        finalpos.write(str(x+49)+' ')  # 加上49,全部正值化
    finalpos.write('\n')
finalpos.close()





