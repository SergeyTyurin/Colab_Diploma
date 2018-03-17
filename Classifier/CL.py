import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (20, 20)
plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_mode_cpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load PASCAL VOC labels
labelmap_file = '/home/styurin/SSD/VGGNet/VOC0712/labelmap_voc.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
    and visualize each (height, width) thing in a grid of size approx.
    sqrt(n) by sqrt(n)"""
    # normalize data for display
    print data.shape
    data = (data - data.min()) / (data.max() - data.min())
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),(0, 1), (0, 1)) + ((0, 0),) * (data.ndim - 3)) # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1) # pad with ones (white)
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data); plt.axis('off')

model_def = '/home/styurin/Diploma/Classifier/deploy_infogain_v3.prototxt'
model_weights = '/home/styurin/Diploma/Classifier/snapshots_v31/CLR_iter_23000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


# set net to batch size of 1
image_resize = 224
net.blobs['data'].reshape(1,3,image_resize,image_resize)

image = caffe.io.load_image('/home/styurin/Dest/Images/6714.png')
#plt.imshow(image)

transformed_image = transformer.preprocess('data', image)
net.blobs['data'].data[...] = transformed_image

# Forward pass.
net.forward()['prob']
print net.blobs['conv1'].data.shape
print net.blobs['conv2'].data.shape
print net.blobs['conv3'].data.shape
print net.blobs['conv4'].data.shape
print net.blobs['conv5'].data.shape
print net.blobs['conv6'].data.shape
print net.blobs['conv7'].data.shape
vis_square(net.blobs['relu5'].data[0])
print net.blobs['prob'].data
plt.show()