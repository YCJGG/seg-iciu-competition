"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
import math
from PIL import Image

import tensorflow as tf
import numpy as np
from  skimage import transform

import pydensecrf.densecrf as dcrf
import matplotlib.pyplot as plt
from scipy import misc
from pydensecrf.utils import compute_unary, create_pairwise_bilateral,create_pairwise_gaussian, softmax_to_unary
from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, prepare_label

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
    
NUM_CLASSES = 2
DATA_DIRECTORY = './val_data'
DATA_LIST_PATH = './val_data/val_c.txt'
SAVE_DIR = './output-3/'
RESTORE_FROM = './10.31_data3_voc/model.ckpt-750'
SXY = 19
SRGB = 6
COMPAT = 5
COMPAT_1 = 3
SXY_1 = 1
TS =7 
def test_acc(gt,crf):
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0
    gtt = gt.reshape(-1)
    crff = crf.reshape(-1)
    for k in range(len(gtt)):
        if gtt[k]==1 and crff[k]== 1:
            tp = tp + 1
        if crff[k]== 1 and gtt[k]== 0:
            fp = fp + 1 
        if crff[k]== 0 and gtt[k]== 0:
            tn = tn + 1
        if crff[k]== 0 and gtt[k]== 1:
            fn = fn + 1
    if (tp+fn)==0:
        tpr = 0
    else:
        tpr = tp/(tp+fn)
    if (fp+tn)==0:
        fpr == 0
    else:
        fpr = fp/(fp+tn)
    return tpr,fpr

def filtered(x1,y1,x2,y2,fraction,width,height):
    xd = x2-x1
    yd = y2-y1
    xfd = xd * fraction
    yfd = yd * fraction
    xx1 = max(0,int(x1-(xfd-xd)/2))
    xx2 = min(int(x2+(xfd-xd)/2),width)
    yy1 = max(0,int(y1-(yfd-yd)/2))
    yy2 = min(int(y2+(yfd-yd)/2),height)
    return (xx1,yy1),(xx2,yy2)


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("img_path", type=str,
                        help="Path to the RGB image file.")
    parser.add_argument("model_weights", type=str, default=RESTORE_FROM,
                        help="Path to the file with model weights.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    parser.add_argument("--sxy", type=int, default=SXY,
                        help="CRF_sxy")
    parser.add_argument("--srgb",type=int,default=SRGB,help = None)
    parser.add_argument("--compat",type=int,default=COMPAT,help = None)
    parser.add_argument("--compat_1",type=int,default=COMPAT_1,help = None)
    parser.add_argument("--sxy_1",type = int,default=SXY_1,help = None)
    parser.add_argument("--ts",type = int ,default=TS,help = None)
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    
    # # Prepare image.
    # img = tf.image.decode_jpeg(tf.read_file(args.img_path), channels=3)
    # # Convert RGB to BGR.
    # img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    # img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # # Extract mean.
    # img -= IMG_MEAN 
    coord = tf.train.Coordinator()
    
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            DATA_DIRECTORY,
            DATA_LIST_PATH,
            None, # No defined input size.
            False, # No random scale.
            False, # No random mirror.
            False,
            255,
            IMG_MEAN,
            coord)
        image, label = reader.image, reader.label
        #image = reader.image
    image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0) # Add one batch dimension.
    #image_batch = tf.expand_dims(image,dim=0)
    # Create network.
    net = DeepLabResNetModel({'data': image_batch}, is_training=False, num_classes=args.num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output_up_1 = tf.image.resize_bilinear(raw_output, tf.shape(image)[0:2,])
    raw_output_up = tf.argmax(raw_output_up_1, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)
    
    probabilities = tf.nn.softmax(raw_output_up_1)
    
    # Set up TF session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, RESTORE_FROM)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    # Perform inference.
    
    file = open('./CRF/val_c.txt')
    file2 = open('./CRF/val_coor.txt')
    coor = []
    for line in file2:
        line = line.strip()
        coor.append(line)

    filenames = []
    for line in file:
        filenames.append(line.strip().split(' ')[1].split('/')[-1])
    #print(filenames)
    tpr_list = []
    fpr_list = []

    file = open('acc_sxy','a')
    for step in range(len(filenames)):
        preds,gt,final_probabilities = sess.run([pred,label_batch,probabilities])
        path = './CRF/val_data_2/'+'ori/'+filenames[step][:-3]+'jpg'
        img = misc.imread(path)
        #preds,img, final_probabilities = sess.run([pred,image,probabilities])
        final_probabilities = final_probabilities.squeeze()        
        final_probabilities = final_probabilities.transpose(2,0,1)
        cor =  coor[step]
        cor = cor.split(' ')
        x1 = int(cor[0])
        x2 = int(cor[2])
        y1 = int(cor[1])
        y2 = int(cor[3])
        #xx = int(cor[2])-int(cor[0])
        #yy = int(cor[3])-int(cor[1])
        
        (x1,y1),(x2,y2) = filtered(x1,y1,x2,y2,1.3,192,256)
        xx = x2 -x1
        yy = y2 - y1
        out_shape = (final_probabilities.shape)
        #print(out_shape)
        #print(img.shape)
        final_probabilities=transform.resize(final_probabilities,(2,yy,xx),preserve_range=True)
        final_probabilities = final_probabilities.astype(np.float32)
        f_shape = (final_probabilities.shape)
        final_probabilities=np.lib.pad(final_probabilities,((0,0),(y1,256-y2),(x1,192-x2)),'constant',constant_values=0.95)
        segPred = -np.log(final_probabilities)
        #print(segPred.shape)
        segPred = np.ascontiguousarray(segPred)
        #print(segPred.shape)
        #print(gt.shape)
        d = dcrf.DenseCRF2D(img.shape[1],img.shape[0], 2)
        d.setUnaryEnergy(segPred.reshape(2,-1))
        #feats = create_pairwise_gaussian(sdims=(5, 5), shape=img.shape[:2])
        #d.addPairwiseEnergy(feats, compat=3,kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)
        
        #feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),img=img, chdim=2)
        #d.addPairwiseEnergy(feats, compat=10,kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)
        d.addPairwiseGaussian(sxy=args.sxy_1, compat=args.compat_1, kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)
        img = img.astype('uint8')
        d.addPairwiseBilateral(sxy=args.sxy, srgb=args.srgb, rgbim=img, compat=args.compat,kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        # inference
        Q = d.inference(args.ts)
        #print(Q.shape)
        #print(segPred.shape)
        segPred = np.argmax(Q, axis=0).reshape(img.shape[0],img.shape[1])
        #print(segPred.shape)
        #break
        #segPred = transform.resize(segPred,(yy,xx),preserve_range=True)
        #print(segPred.shape,x1,x2,y1,y2)
        #break
        #segPred = np.resize(segPred,(yy,xx))
        #print(segPred)
        #break
        #segPred = np.lib.pad(segPred,((y1,256-y2),(x1,192-x2)),'constant',constant_values = 0)
        #print(segPred.shape)
        #print(gt.shape)
        #break
        #segPred[segPred == 1]=128
        segPred = segPred.astype('int')        
        msk = decode_labels(segPred[np.newaxis,:,:,np.newaxis], num_classes=2)
        segPred = segPred.astype('uint8')
        #msk = segPred
        #lab = decode_labels(gt, num_classes=2)
        #pr = decode_labels(preds,num_classes=2)
        im = Image.fromarray(msk[0])
        #la = Image.fromarray(lab[0])
        #pr = Image.fromarray(pr[0])
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        #np.save(args.save_dir+str(step)+'.npy',final_probabilities)
        im.save(args.save_dir +str(step)+ 'crf_'+filenames[step])
        #break
        #la.save(args.save_dir +'gt_'+ filenames[step])
        #pr.save(args.save_dir+'pred_'+ filenames[step])
        #print('The output file has been saved to {}'.format(args.save_dir +str(step)+ '_mask.png'))
        tpr,fpr = test_acc(gt,segPred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        #break
    tpr_aver = np.mean(tpr_list)
    fpr_aver = np.mean(fpr_list)
    print(tpr_aver,fpr_aver)
    acc = math.sqrt((1-tpr_aver)**2+fpr_aver**2)
    acc = round(acc,4)
    
    print(str(args.sxy)+' '+str(args.srgb)+' '+str(args.compat)+' '+str(args.compat_1)+' '+str(args.sxy_1)+' '+str(args.ts)+' '+'final acc: ',acc)
    file.write(str(args.sxy)+' ' +str(args.srgb)+' '+str(args.compat)+' '+str(args.compat_1)+' '+str(args.sxy_1)+' '+str(args.ts)+' '+str(acc)+'\n')
    coord.request_stop()
    coord.join(threads)
        
if __name__ == '__main__':
    main()
