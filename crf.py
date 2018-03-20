import os
sxy = 0 
while(sxy<=9):
    sxy = sxy + 1 
    command = 'CUDA_VISIBLE_DEVICES=1 '+'python '+'inference_crf.py '+'./1 '+'./1 '+'--ts='+str(sxy)
    os.system(command)
