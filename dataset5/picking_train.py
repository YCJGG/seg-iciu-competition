import os
file=open('train.txt','a')
ori_image=[]
gt = [] 
for root,dir,filenames in os.walk('./1'):
    ori_image = filenames
    for k in range(len(ori_image)):
        file.write('/1/'+ori_image[k]+' '+'/2/'+ori_image[k][0:-3]+'png'+'\n')
        #file.write('\n')