import os

i=0
lr = 1e-4

while(i<3): 
    k = 0
    lr = lr*0.3    
    fine_turen_order = 'python '+'train.py '+'--not-restore-last '+ '--snapshot-dir=./snapshots_finetune-' + str(i) +'/ ' +'--learning-rate='+str(lr)
    os.system(fine_turen_order)
    
    while(k<10001):
        order = 'python '+'evaluate.py '+'--restore-from '+'./snapshots_finetune-'+str(i)+str(i)+'/'+'model.ckpt-'+str(k)
        os.system(order)
        k=k+100
   
    i = i+1

