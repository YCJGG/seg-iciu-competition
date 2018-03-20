import os
k = 0
while(k<1501):
    order = 'python '+'evaluate.py '+'--restore-from '+'./snapshots_finetune_lr_decay-00'+'/'+'model.ckpt-'+str(k)
    os.system(order)
    k=k+25