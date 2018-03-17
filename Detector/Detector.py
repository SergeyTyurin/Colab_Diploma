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
labelmap_file = '/home/styurin/Diploma/lmdb/label_map_s.prototxt'
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
    plt.show()

model_def = '/home/styurin/Diploma/Detector/deploy.prototxt'
model_weights = '/home/styurin/Diploma/Detector/snapshots_v1/DTR_iter_14000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


# set net to batch size of 1
image_resize = 300
net.blobs['data'].reshape(1,3,image_resize,image_resize)

image = caffe.io.load_image('/home/styurin/Dest/Images/6345.png')
#plt.imshow(image)

transformed_image = transformer.preprocess('data', image)
net.blobs['data'].data[...] = transformed_image

# Forward pass.
detections = net.forward()['detection_out']
# print net.blobs['conv4_3'].data.shape[1:]
# new_shape = net.blobs['conv4_3'].data.shape[1:]
# net.blobs['conv4_3'].data.reshape(new_shape)
vis_square(net.blobs['conv7_norm_mbox_conf'].data[0])
# print len(net.blobs['conv4_3'].data[0])
# print len(net.blobs['conv4_3'].data[0][0])
# print len(net.blobs['conv4_3'].data[0][0][0])
# print net.blobs['conv4_3_norm_mbox_conf_perm'].data
# print len(net.blobs['conv4_3_norm_mbox_conf_perm'].data[0])
# print len(net.blobs['conv4_3_norm_mbox_conf_perm'].data[0][0])
# print len(net.blobs['conv4_3_norm_mbox_conf_perm'].data[0][0][0])

# print net.blobs['conv4_3_norm_mbox_conf_flat'].data
# print len(net.blobs['conv4_3_norm_mbox_conf_flat'].data[0])
#
# print net.blobs['conv4_3_norm_mbox_priorbox'].data
#
# print net.blobs['mbox_priorbox'].data
#
# print '\n'
# print net.blobs['mbox_conf'].data, len(net.blobs['mbox_conf'].data[0])
#
# print '\n'
# print net.blobs['mbox_conf_reshape'].data, len(net.blobs['mbox_conf_reshape'].data[0])
#
# print '\n'
# print net.blobs['mbox_conf_softmax'].data, len(net.blobs['mbox_conf_softmax'].data[0])
#
# print '\n'
# print net.blobs['mbox_conf_flatten'].data, len(net.blobs['mbox_conf_flatten'].data[0])

# Parse the outputs.
det_label = detections[0,0,:,1]
det_conf = detections[0,0,:,2]
det_xmin = detections[0,0,:,3]
det_ymin = detections[0,0,:,4]
det_xmax = detections[0,0,:,5]
det_ymax = detections[0,0,:,6]

# Get detections with confidence higher than 0.6.
top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

top_conf = det_conf[top_indices]
top_label_indices = det_label[top_indices].tolist()
top_labels = get_labelname(labelmap, top_label_indices)
top_xmin = det_xmin[top_indices]
top_ymin = det_ymin[top_indices]
top_xmax = det_xmax[top_indices]
top_ymax = det_ymax[top_indices]

print top_conf
print top_label_indices
print top_labels
print top_xmin, top_xmax
print top_ymin, top_ymax

colors = plt.cm.hsv(np.linspace(0, 1, 4)).tolist()

currentAxis = plt.gca()

for i in xrange(top_conf.shape[0]):
    xmin = int(round(top_xmin[i] * image.shape[1]))
    ymin = int(round(top_ymin[i] * image.shape[0]))
    xmax = int(round(top_xmax[i] * image.shape[1]))
    ymax = int(round(top_ymax[i] * image.shape[0]))
    score = top_conf[i]
    label = int(top_label_indices[i])
    label_name = top_labels[i]
    display_txt = '%s: %.2f'%(label_name, score)
    coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
    color = colors[label]
    currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
    currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})

plt.imshow(image)
plt.show()
