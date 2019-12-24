import os


def findbegin(infile):
    while True:
        strxml = infile.readline()
        # print(strxml)
        print("ready to read begin")
        i = 0
        while i<len(strxml)-7:
            print(strxml[i:i+8])
            if strxml[i:i+8]=="<weibos>":
                return True
            i = i+1
    return False


def fileconvert(targetpath, i_filename, targettype):
    infile = open(targetpath + "/" + i_filename + targettype, "r", encoding='utf-8')
    outfile = open(i_filename + ".txt", "w", encoding='utf-8')

    if not findbegin:
        outfile.write("--")
        infile.close()
        outfile.close()
        return

    curid = 0
    text = 0
    while True:
        strxml = infile.readline()
        i = 0
        while i < len(strxml):
            if i < len(strxml) - 7 and strxml[i:i + 8] == "</weibo>":  # read weibo end
                text = 0
                outfile.write("\n")
                break
            if text == 1 and strxml[i]!=' ' and strxml[i]!='\n':
                outfile.write(strxml[i])
            # print(strxml[i:i + 9])
            if i < len(strxml)-8 and strxml[i:i + 9] == "</weibos>":
                outfile.write("--")
                infile.close()
                outfile.close()
                return
            elif i < len(strxml)-9 and strxml[i:i+10] == "<weibo id=":   # read weibo begin
                i = i + 11
                num = 0
                while strxml[i].isdigit():         # read weibo id num
                    num = num*10 + int(strxml[i])
                    i = i + 1
                curid = num
                text = 1
                outfile.write(str(curid)+"\n")
                i = i + 1                        # read">"
            i = i + 1
    return


def xtt():
    targetpath = "C:/Users/Leqsott/PycharmProjects/Employment-Relation-Extraction/testfile"
    targettype = ".xml"
    filelist = os.listdir(targetpath)
    for i in filelist:
        i_fullname = os.path.splitext(i)
        i_filename = i_fullname[0]  # 读取文件名
        i_filetype = i_fullname[1]  # 读扩展名
        if i_filename == "task1_input" and i_filetype == targettype:  # 如果和目标扩展名匹配
            fileconvert(targetpath, i_filename, targettype)
