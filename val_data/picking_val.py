import os
file=open('val.txt','a')
ori_image=[]
gt = [] 
for root,dir,filenames in os.walk('./ori'):
    ori_image = filenames
    for k in range(len(ori_image)):
        file.write('/ori/'+ori_image[k]+' '+'/sil/'+ori_image[k][0:-3]+'png'+'\n')
        #file.write('\n')