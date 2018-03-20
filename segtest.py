import numpy as np
from skimage.transform import resize
import tensorflow as tf
from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, prepare_label
import time
from scipy import misc
import matplotlib.pyplot as plt

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
RESTORE_FROM = './10.31_data3_voc/model.ckpt-750'

time1 = time.time()

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

# network set up
tmp = tf.placeholder(tf.float32,[1,256,None,3])
#tmp=tf.convert_to_tensor(np.zeros([256,186, 3]).reshape(1,256,186,3).astype(np.float32))
net = DeepLabResNetModel({'data':tmp}, is_training=False, num_classes=2)
raw_output = net.layers['fc1_voc12']
raw_output_up=tf.image.resize_bilinear(raw_output,tf.shape(tmp)[1:3,])
raw_output_up=tf.argmax(raw_output_up,dimension=3)
pred = tf.expand_dims(raw_output_up,dim=3)
#raw_output_up = tf.nn.softmax(raw_output)

debug = net.layers['conv1']

# Set up TF session and initialize variables
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Load weights
loader = tf.train.Saver()
loader.restore(sess, RESTORE_FROM)

img = misc.imread('0081.jpg')
imagePatch = resize(img, [128, int(128./img.shape[0]*img.shape[1])])[:,:,::-1]
resized_imagePatch = resize(imagePatch, img.shape).astype(np.float32)-IMG_MEAN
resized_imagePatch = resized_imagePatch[np.newaxis,:,:,:]
#resized_imagePatch = tf.convert_to_tensor(resized_imagePatch)

#pred,final_probabilities = sess.run([raw_output,raw_output_up], feed_dict={tmp: resized_imagePatch})
preds = sess.run(pred,feed_dict={tmp:resized_imagePatch})
print(preds.shape)
predd = preds.squeeze()
#final_probabilities = final_probabilities.squeeze()
#np.save('1.npy',pred[:,:,0])
#prob = resize(final_probabilities, [img.shape[0], img.shape[1]], preserve_range=True)#.transpose(2,0,1)

#print final_probabilities.shape
#print final_probabilities
print (predd.shape)
misc.imsave('1.png',predd)
#misc.imsave('2.png',predd[:,:,1])

print time.time()-time1
