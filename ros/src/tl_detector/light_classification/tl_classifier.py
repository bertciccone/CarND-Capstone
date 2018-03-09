from styx_msgs.msg import TrafficLight

# tl_classifier_imports
import pickle
#import numpy as np
from numpy import zeros, newaxis
import cv2
#import time
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelBinarizer
from keras.models import load_model
import h5py
import keras
#from keras.models import Model, Sequential
#from keras.layers.core import Dense, Activation, Flatten
#from keras.models import Model, Sequential
#from keras.layers import Input, Activation, Concatenate, Dense
#from keras.layers import Flatten, Dropout, Reshape
#from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D
#from keras.layers import AveragePooling2D

#from keras.optimizers import SGD,RMSprop,adam
#from keras.utils import np_utils

#import matplotlib.pyplot as plt
#import matplotlib
import os
#import tensorflow as tf
#from PIL import Image

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

import skimage
import skimage.io as io


# CarND Object Detection Lab

import tensorflow as tf
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
import time
from scipy.stats import norm

#%matplotlib inline
plt.style.use('ggplot')

# Exercise 1 - Implement Separable Depthwise Convolution

def vanilla_conv_block(x, kernel_size, output_channels):
    """
    Vanilla Conv -> Batch Norm -> ReLU
    """
    x = tf.layers.conv2d(
        x, output_channels, kernel_size, (2, 2), padding='SAME')
    x = tf.layers.batch_normalization(x)
    return tf.nn.relu(x)

# TODO: implement MobileNet conv block
def mobilenet_conv_block(x, kernel_size, output_channels):
    """
    Depthwise Conv -> Batch Norm -> ReLU -> Pointwise Conv -> Batch Norm -> ReLU
    """
    # assumes BHWC format
    input_channel_dim = x.get_shape().as_list()[-1]
    W = tf.Variable(tf.truncated_normal((kernel_size, kernel_size, input_channel_dim, 1)))

    # depthwise conv
    x = tf.nn.depthwise_conv2d(x, W, (1, 2, 2, 1), padding='SAME')
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)


    # pointwise conv
    x = tf.layers.conv2d(x, output_channels, (1, 1), padding='SAME')
    x = tf.layers.batch_normalization(x)

    return tf.nn.relu(x)

# constants but you can change them so I guess they're not so constant :)
INPUT_CHANNELS = 32
OUTPUT_CHANNELS = 512
KERNEL_SIZE = 3
IMG_HEIGHT = 256
IMG_WIDTH = 256

with tf.Session(graph=tf.Graph()) as sess:
    # input
    x = tf.constant(np.random.randn(1, IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS), dtype=tf.float32)

    with tf.variable_scope('vanilla'):
        vanilla_conv = vanilla_conv_block(x, KERNEL_SIZE, OUTPUT_CHANNELS)
    with tf.variable_scope('mobile'):
        mobilenet_conv = mobilenet_conv_block(x, KERNEL_SIZE, OUTPUT_CHANNELS)

    vanilla_params = [
        (v.name, np.prod(v.get_shape().as_list()))
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'vanilla')
    ]
    mobile_params = [
        (v.name, np.prod(v.get_shape().as_list()))
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'mobile')
    ]

    print("VANILLA CONV BLOCK")
    total_vanilla_params = sum([p[1] for p in vanilla_params])
    for p in vanilla_params:
        print("Variable {0}: number of params = {1}".format(p[0], p[1]))
    print("Total number of params =", total_vanilla_params)
    print()

    print("MOBILENET CONV BLOCK")
    total_mobile_params = sum([p[1] for p in mobile_params])
    for p in mobile_params:
        print("Variable {0}: number of params = {1}".format(p[0], p[1]))
    print("Total number of params =", total_mobile_params)
    print()

    print("{0:.3f}x parameter reduction".format(total_vanilla_params /
                                             total_mobile_params))

# Object Detection Inference

# Frozen inference graph files. NOTE: change the path to where you saved the models.
#SSD_GRAPH_FILE = '/Users/Bert/Projects/Udacity/CARND/term3/quizzes/18-scene-object-detection-lab/ModelZoo/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
SSD_GRAPH_FILE = './frozen_inference_graph.pb'
RFCN_GRAPH_FILE = '/Users/Bert/Projects/Udacity/CARND/term3/quizzes/18-scene-object-detection-lab/ModelZoo/rfcn_resnet101_coco_2017_11_08/frozen_inference_graph.pb'
FASTER_RCNN_GRAPH_FILE = '/Users/Bert/Projects/Udacity/CARND/term3/quizzes/18-scene-object-detection-lab/ModelZoo/faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08/frozen_inference_graph.pb'

print("Number of colors/classes, one color per class")

# Colors (one for each class)
cmap = ImageColor.colormap
print("Number of colors =", len(cmap))
COLOR_LIST = sorted([c for c in cmap.keys()])

#
# Utility funcs
#

def filter_boxes(min_score, boxes, scores, classes):
    """Return boxes with a confidence >= `min_score`"""
    n = len(classes)
    idxs = []
    for i in range(n):
        if scores[i] >= min_score:
            idxs.append(i)

    filtered_boxes = boxes[idxs, ...]
    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    return filtered_boxes, filtered_scores, filtered_classes

def to_image_coords(boxes, height, width):
    """
    The original box coordinate output is normalized, i.e [0, 1].

    This converts it back to the original coordinate based on the image
    size.
    """
    box_coords = np.zeros_like(boxes)
    box_coords[:, 0] = boxes[:, 0] * height
    box_coords[:, 1] = boxes[:, 1] * width
    box_coords[:, 2] = boxes[:, 2] * height
    box_coords[:, 3] = boxes[:, 3] * width

    return box_coords

def draw_boxes(image, boxes, classes, thickness=4):
    """Draw bounding boxes on the image"""
    draw = ImageDraw.Draw(image)
    for i in range(len(boxes)):
        bot, left, top, right = boxes[i, ...]
        class_id = int(classes[i])
        color = COLOR_LIST[class_id]
        draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)

def load_graph(graph_file):
    """Loads a frozen inference graph"""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph

detection_graph = load_graph(SSD_GRAPH_FILE)
#detection_graph = load_graph(RFCN_GRAPH_FILE)
#detection_graph = load_graph(FASTER_RCNN_GRAPH_FILE)

# The input placeholder for the image.
# `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Each box represents a part of the image where a particular object was detected.
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represent how level of confidence for each of the objects.
# Score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

# The classification of the object (integer id).
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Load a sample image.
image = Image.open('assets/sample1.jpg')
image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

with tf.Session(graph=detection_graph) as sess:
    # Actual detection.
    (boxes, scores, classes) = sess.run([detection_boxes, detection_scores, detection_classes],
                                        feed_dict={image_tensor: image_np})

    # Remove unnecessary dimensions
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes)

    confidence_cutoff = 0.8
    # Filter boxes with a confidence score less than `confidence_cutoff`
    boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)

    # The current box coordinates are normalized to a range between 0 and 1.
    # This converts the coordinates actual location on the image.
    width, height = image.size
    box_coords = to_image_coords(boxes, height, width)

    # Each class with be represented by a differently colored box
    draw_boxes(image, box_coords, classes)

    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.savefig('assets/dog_boxes.jpg')

# Timing Detection

def time_detection(sess, img_height, img_width, runs=10):
    image_tensor = sess.graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = sess.graph.get_tensor_by_name('detection_scores:0')
    detection_classes = sess.graph.get_tensor_by_name('detection_classes:0')

    # warmup
    gen_image = np.uint8(np.random.randn(1, img_height, img_width, 3))
    sess.run([detection_boxes, detection_scores, detection_classes], feed_dict={image_tensor: gen_image})

    times = np.zeros(runs)
    for i in range(runs):
        t0 = time.time()
        sess.run([detection_boxes, detection_scores, detection_classes], feed_dict={image_tensor: image_np})
        t1 = time.time()
        times[i] = (t1 - t0) * 1000
    return times

with tf.Session(graph=detection_graph) as sess:
    times = time_detection(sess, 600, 1000, runs=10)

print("Detection Times")
print(times)

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.model = load_model('AVO4_model.h5')
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()

        print ('Loaded Model')
#        self.model.summary()
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        desired_dim=(32,32)
        img_resized = cv2.resize(image, desired_dim, interpolation=cv2.INTER_LINEAR)
        img_ = np.expand_dims(np.array(img_resized), axis=0)

        with self.graph.as_default():
           predict_tl_state = self.model.predict_classes(img_)
#           print ('predict_tl_state = ', predict_tl_state)
        if predict_tl_state == 0:
           print ('Traffic Light Color = RED ', predict_tl_state[0] )
           return TrafficLight.RED
        elif predict_tl_state == 1:
           print ('Traffic Light Color = YELLOW ', predict_tl_state[0])
           return TrafficLight.YELLOW
        elif predict_tl_state == 2:
           print ('Traffic Light Color = GREEN ',  predict_tl_state[0])
           return TrafficLight.GREEN
        else:
           return TrafficLight.UNKNOWN


