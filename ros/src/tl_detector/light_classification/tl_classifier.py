from styx_msgs.msg import TrafficLight

import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
import cv2

import rospy

DETECTION_MODEL="models/ssd_inception_{}.pb"
CLASSIFY_MODEL="models/classifier_{}.h5"

DETECTION_MIN_SCORE = 0.70

DEBUG_OUTPUT = 1        # 0: No output 1: important message 2:all

CLASS_TO_TL = [
    TrafficLight.RED,
    TrafficLight.YELLOW,
    TrafficLight.GREEN,
    TrafficLight.UNKNOWN
]

TL_TO_TEXT = {
    TrafficLight.RED: "RED",
    TrafficLight.YELLOW: "YELLOW",
    TrafficLight.GREEN: "GREEN",
    TrafficLight.UNKNOWN: "UNKNOWN"
}

class TLClassifier(object):
    def __init__(self, env = 'sim'):
        self.log_info('*** creating classifier for environment "{}"'.format(env))
        # load the detection model
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(DETECTION_MODEL.format(env), 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        
            self.input_image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.output_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.output_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.output_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.output_num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
    
        # self.detection_session = tf.Session(graph=self.detection_graph)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
        self.detection_session = tf.Session(graph=self.detection_graph, config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        # load the classifier
        self.classifier_session = tf.Session()
        K.set_session(self.classifier_session)
        self.classifier_model = load_model(CLASSIFY_MODEL.format(env))
        self.classifier_model._make_predict_function()
        self.classifier_graph = tf.get_default_graph()

        # prime the detection model
        self.detect_traffic_lights(np.zeros((64, 32, 3)))

    def detect_traffic_lights(self, image):
        # bounding box detection.
        with self.detection_graph.as_default():
            # expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(image, axis=0)  
            (boxes, scores, classes, num) = self.detection_session.run(
                [self.output_boxes, self.output_scores, self.output_classes, self.output_num_detections],
                feed_dict={self.input_image_tensor: img_expanded})
            
        # all outputs are float32 numpy arrays, so convert types as appropriate
        num = int(num[0])
        classes = classes[0].astype(np.uint8)
        boxes = boxes[0]
        scores = scores[0]

        # only keep high confidence traffic light detections
        mask = (scores >= DETECTION_MIN_SCORE) & (classes == 10)
        boxes, scores, classes = boxes[mask], scores[mask], classes[mask]

        return boxes, scores, classes

    def classify_detected_traffic_lights(self, image, boxes):
        if (len(boxes) <= 0):
            self.log_info('UNKNOWN LIGHT (no boxes)')
            return TrafficLight.UNKNOWN

        lights = []

        for i in range(len(boxes)):
            # convert to image coordinates
            top, left, bottom, right = int(boxes[i][0] * image.shape[0]), \
                                       int(boxes[i][1] * image.shape[1]), \
                                       int(boxes[i][2] * image.shape[0]), \
                                       int(boxes[i][3] * image.shape[1])
        
            # extract light
            light = image[top:bottom, left:right]
            light = cv2.resize(light, (32, 64))
            light = light / 255.0
            lights.append(light)

        lights = np.array(lights)
        with self.classifier_graph.as_default():
            results = self.classifier_model.predict_on_batch(lights)

        results = results.sum(axis=0)
        class_id = np.argmax(results[0:3])      # ignore unknown

        if (results[class_id] == 0):
            return TrafficLight.UNKNOWN

        return CLASS_TO_TL[class_id]

    def log_info(self, msg):
        if DEBUG_OUTPUT > 0:
            rospy.loginfo(msg)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        ts_before = rospy.Time.now()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, scores, classes = self.detect_traffic_lights(image)
        detected_class =  self.classify_detected_traffic_lights(image, boxes)
        ts_after = rospy.Time.now()

        self.log_info('Detected light = {} (duration = {})'.format(TL_TO_TEXT[detected_class], (ts_after - ts_before).to_sec() * 1000))

        return detected_class