import os

i=2
lr = 2.5e-4
p = 0

while(p<2):
    k = 0
    while(k<5001):
        order = 'python '+'evaluate.py '+'--restore-from '+'./snapshots_finetune-'+str(p)+str(p)+'/'+'model.ckpt-'+str(k)
        os.system(order)
        k=k+100
    p = p + 1
while(i<4): 
    k = 0
    lr = lr/(2.5*(i+1))
    fine_turen_order = 'python '+'train.py '+'--not-restore-last '+ '--snapshot-dir=./snapshots_finetune-' + str(i) +'/ ' +'--learning-rate='+str(lr)
    os.system(fine_turen_order)
    while(k<5001):
        order = 'python '+'evaluate.py '+'--restore-from '+'./snapshots_finetune-'+str(i)+str(i)+'/'+'model.ckpt-'+str(k)
        os.system(order)
        k=k+100
   
    i = i+1