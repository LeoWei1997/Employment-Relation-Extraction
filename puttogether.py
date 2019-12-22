# 文件合体
filePath=['./LabeledData.%d.txt'%i for i in range(1,6)]
LabeledData=open('./LabeledData.txt','w',encoding='utf-8')

for file in filePath:
    f=open(file,'r',encoding='utf-8')
    for sent in f:
        LabeledData.write(sent)
LabeledData.close()