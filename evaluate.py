"""Evaluation script for the DeepLab-ResNet network on the validation subset
   of PASCAL VOC dataset.
This script evaluates the model on 1449 validation images.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
import math
import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel, ImageReader, prepare_label

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = './val_data'
DATA_LIST_PATH = './val_data/val.txt'
IGNORE_LABEL = 255
NUM_CLASSES = 2
NUM_STEPS = 100 # Number of images in the validation set.
RESTORE_FROM = './snapshots_finetune/model.ckpt-19500'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of images in the validation set.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
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
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            None, # No defined input size.
            False, # No random scale.
            False, # No random mirror.
            False,
            args.ignore_label,
            IMG_MEAN,
            coord)
        image, label = reader.image, reader.label
    image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0) # Add one batch dimension.

    # Create network.
    net = DeepLabResNetModel({'data': image_batch}, is_training=False, num_classes=args.num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()
    
    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output = tf.argmax(raw_output, dimension=3)
    pred = tf.expand_dims(raw_output, dim=3) # Create 4-d tensor.
    
    # acc
    pred = tf.reshape(pred, [-1,])
    gt = tf.reshape(label_batch, [-1,])
    #miou
    weights = tf.cast(tf.less_equal(gt, args.num_classes-1), tf.int32) # Ignoring all labels greater than or equal to n_classes.
    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=args.num_classes, weights=weights)
    
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    if args.restore_from is not None:
        load(loader, sess, args.restore_from)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    # Iterate over training steps.
    tpr_list  = []
    fpr_list  = []
    #miou_list = np.zeros(100)
    #cur_miou = 0.0
    for step in range(args.num_steps):
        tp = 0.0
        fp = 0.0
        tn = 0.0
        fn = 0.0
        preds, gtt,_= sess.run([pred,gt,update_op])
        #img , lab = sess.run([image_batch, label_batch])
        #img = np.array(img)
        #lab = np.array(lab)
        #print(img.shape,lab.shape)
        
        #miou_list[step] = res[0]
        pr = np.array(preds)
        gtt = np.array(gtt)
        #print(len(pr))
        #print(len(gtt))
        for k in range(len(gtt)):
            if pr[k] == 1 and gtt[k]== 1:
                tp = tp + 1
            if pr[k] == 1 and gtt[k]==0:
                fp = fp + 1
            if pr[k] == 0 and gtt[k] == 0:
                tn = tn + 1
            if pr[k] == 0 and gtt[k] == 1:
                fn = fn + 1

        tpr = tp/(tp+fn)
        fpr = fp/(fp+tn)
        if step % 50 == 0:
            print('step {:d}'.format(step))
            print('len(gt): ',len(gtt))
            print('tp: ',tp)
            print('fp: ',fp)
            print('tn:',tn)
            print('fn:',fn)
            print(tpr,fpr)
            print('\n')
            
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    tpr_aver = np.mean(tpr_list)
    fpr_aver = np.mean(fpr_list)
    acc = math.sqrt((1-tpr_aver)**2+fpr_aver**2)
    acc = round(acc,4)
    print('final acc: ',acc)
    print('Mean IoU: {:.3f}'.format(mIoU.eval(session=sess)))
    cur_miou = mIoU.eval(session=sess)
    miou = round(cur_miou,4)
    file = open('miou_acc','a')
    file.write(str(acc)+' '+str(miou)+'\n')

    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
