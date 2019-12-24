import os


def getxml(targetpath, filename, targettype):
    # infile = open(targetpath + "/" + i_filename + ".txt", "r", encoding='utf-8')
    # relafile = open(targetpath + "/" + i_filename + targettype, "r", encoding='utf-8')

    infile = open(filename + ".txt", "r", encoding='utf-8')
    relafile = open(filename + targettype, "r", encoding='utf-8')
    outfile = open("test1.xml", "w", encoding='utf-8')

    outfile.write("<weibos>\n")
    while True:
        strtxt = infile.readline()   # read weibo id
        if strtxt.strip() == "--":
            break
        outfile.write("\t<weibo id=\""+strtxt.strip()+"\">\n")  # write weibo id
        outfile.write("\t\t"+infile.readline())  # read and write the content

        while True: # read relations
            strrela0 = relafile.readline()
            if strrela0.strip().isdigit():
                break
            strrela1 = relafile.readline()
            outfile.write("\t\t<employee from=\""+ strrela0.strip() +"\" name=\"" + strrela1.strip() + "\"></employee>\n")

        outfile.write("\t</weibo>\n")

    outfile.write("</weibos>")
    infile.close()
    relafile.close()
    outfile.close()

    return


def ttx():
    targetpath = "C:/Users/Leqsott/PycharmProjects/Employment-Relation-Extraction/testfile"
    targettype = ".rela"
    filelist = os.listdir(targetpath)
    for i in filelist:
        i_fullname = os.path.splitext(i)
        i_filename = i_fullname[0]  # 读取文件名
        i_filetype = i_fullname[1]  # 读扩展名
        if i_filename == "task1_input" and i_filetype == targettype:  # 如果和目标扩展名匹配
            getxml(targetpath, i_filename, targettype)


