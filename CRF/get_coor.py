import numpy as np
file = open('val_ori.txt')
ori_pic = []
for line in file:
    line = line.strip()[-8:]
    ori_pic.append(line)
file2 = open('coor.txt','r')
index = []    
content=[]
content1 = []
for line in file2:
    line  = line.strip()
    content1.append(line)
    content.append(line[-8:])
for it in ori_pic:
    a = content.index(it)
    index.append(a)
file3 = open('val_coor.txt','a')
file4 = open('val_c.txt','a')
for ind in index:
    file3.write(str(content1[ind+1])+'\n')
    file4.write('/ori/'+str(content1[ind][-8:])+' '+'/sil/'+str(content1[ind][-8:-3])+'png'+'\n')
