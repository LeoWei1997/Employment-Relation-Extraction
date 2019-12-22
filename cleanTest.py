# -*- coding: utf-8 -*-
# 多看看训练集,删掉过长的句子
# 这个主要是文本清理编号,关系对生成(有正有反)
filePath='./LabeledData.txt'
sentList=[]
relationList=[]
NE=[]  # 命名实体,辅助分词
data=open(filePath,'r',encoding='utf-8').readlines()
i=len(data)
j=0
import re
kickout = re.compile('([０-９]*日电)|([０-９]*[年日月时分])|[０-９0-9，、。（）〈〉‘’“”：∶；！?？％《》『』{}●/／nrtsz\n×’．…了]')  # ,居然有人叫张日 能不能去死,正则要有先后顺序  Ｂ
sentid=-1
relationpositive=open('./relation_positive.txt','w',encoding='utf-8')
while j<i:
    sentid += 1  # 句子编号
    sent=re.split('[，。：；]', data[j])  # 根据,.:分割句子
    # print(sent)
    relationNE_nr=[]  # 人名
    relationNE_nt=[]  # 机构名
    idx=0
    while idx<len(sent):
        subsent=sent[idx]
    # for subsent in sent:  # 遍历时删除会导致索引前移然后出错
    #     print(subsent)
        subnr=subnt=None
        try:
            subnr=re.findall('{[^{]*/nr}',subsent)  # {.*/nr}出错了,两个括号我擦...{北京/ns}１２月３０日讯{新华社/nt}
        except:
            pass
        try:
            subnt = re.findall('{[^{]*/nt}', subsent)
        except:
            pass
        if subnt:  # 有机构
            new=False  # 机构去重
            for nt in subnt:
                if nt not in relationNE_nt:
                    new=True
                    # print('find nt ',nt)
                    relationNE_nt.append(nt)
                    if nt not in NE:
                        NE.append(nt)
            if subnr:
                for nr in subnr:
                    if nr not in relationNE_nr:
                        relationNE_nr.append(nr)
                        # print('find nr ', nr)
                        if nr not in NE:
                            NE.append(nr)
            elif not new:  # 没有新机构,没人名
                sent.remove(subsent)
                idx-=1
        elif subnr:  # 没机构有人名
            new =False
            for nr in subnr:
                if nr not in relationNE_nr:
                    new =True
                    relationNE_nr.append(nr)
                    # print('find nr ', nr)
                    if nr not in NE:
                        NE.append(nr)
            if not new:  # 人名已出现过,丢弃
                # print('丢弃')
                sent.remove(subsent)
                idx-=1
        else:  # 没机构没人名,丢弃
            # print('丢弃')
            sent.remove(subsent)
            idx-=1
        idx+=1

    # 清洗符号
    # for subsent in sent:
    #     print(subsent,)
    #     subsent=re.sub(kickout,'',subsent)  # 并没有改变!!! 需要用下标
    #     print('清洗后 ',subsent)
    # for nr in relationNE_nr:
    #     nr=re.sub(kickoutfornrt,'',nr)
    # for nt in relationNE_nt:
    #     nt=re.sub(kickoutfornrt,'',nt)
    idx = 0
    while idx < len(sent):
        sent[idx] = re.sub(kickout, '', sent[idx])
        idx+=1
    idx=0
    while idx< len(relationNE_nt):
        relationNE_nt[idx]=re.sub(kickout,'',relationNE_nt[idx])
        idx+=1
    idx = 0
    while idx < len(relationNE_nr):
        relationNE_nr[idx] = re.sub(kickout, '', relationNE_nr[idx])
        idx+=1
    idx=0
    while idx<len(NE):
        NE[idx]=re.sub(kickout,'',NE[idx])
        idx+=1
    # 添加句子
    sentList.append(sent)

    # 添加关系,需要先处理组合一下
    # print('j++')
    j+=1
    # 读取此句标注的关系
    sentrelation=[]
    while j<i and data[j] != '\n':
        nrnt=data[j].split('|')[:2]  # ['胡晓梦', '新华社'] 后面不管是IEB反正都是雇员
        nr=nrnt[0].split(',')  # 孙奇逢,孙先生
        nt=nrnt[1].split(',')  # 国务院侨务办公室,中华人民共和国国务院侨务办公室
        for r in nr:
            for t in nt:
                sentrelation.append([r,t])
        j+=1
    j+=1

    # print('cur new ',sent, 'curid ',sentid)  # 我擦\n\n读取的时候视为一行,去掉一个还有一个\n!!!!!!导致id混乱

    # 开始下一条新闻前先组合本条新闻的正例和反例
    for nr in relationNE_nr:
        for nt in relationNE_nt:
            if [nr, nt] in sentrelation:
                relationList.append([sentid, nr, nt, 1])
                relationpositive.write('%d %s %s %d\n'%(sentid,nr,nt,1))
            else:
                relationList.append([sentid, nr, nt, 0])
relationpositive.close()
cleansentRes=open('./cleanNews.txt','w',encoding='utf-8')
cleanrelaRes=open('./relation_pos_neg.txt','w',encoding='utf-8')
for sent in sentList:
    for subsent in sent:
        cleansentRes.write(subsent)
    cleansentRes.write('\n')
for rela in relationList:
    for item in rela:
        cleanrelaRes.write(str(item)+' ')
    cleanrelaRes.write('\n')
cleanrelaRes.close()
cleansentRes.close()
NEfile=open('./NE.txt','w',encoding='utf-8')
for ne in NE:
    NEfile.write(ne+'\n')
NEfile.close()








