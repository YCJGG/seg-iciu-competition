import os
k = 0
while(k<1001):
    order = 'python '+'evaluate.py '+'--restore-from '+'./snapshots_finetune-00'+'/'+'model.ckpt-'+str(k)
    os.system(order)
    k=k+25
