import cv2
import numpy as np
from styx_msgs.msg import TrafficLight
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf

class TLClassifier(object):
    def __init__(self):
#        self.model = self.network()
        self.model = load_model('model.h5') 
	self.model._make_predict_function()
        self.graph = tf.get_default_graph()

        #print(model)
        pass
    def network():
        """
        Define the network
        :return:
        """
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(4))
        model.add(Activation('softmax'))

        return model

    def test_an_image(self,image):
        """
        resize the input image to [32, 32, 3], then feed it into the NN for prediction
        :param file_path:
        :return:
        """

        desired_dim=(32,32)
        img_resized = cv2.resize(image, desired_dim, interpolation=cv2.INTER_LINEAR)
        img_ = np.expand_dims(np.array(img_resized), axis=0)

        predicted_state = self.model.predict_classes(img_)

        return predicted_state


    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
#        TESTING_SIMPLE = False
        TESTING_SIMPLE = True
        RED_SAFETY = True


        ####################################################################
        ####################################################################

        if TESTING_SIMPLE:
          # Transform to HSV and simply count the number of color within the range 
 	  hsv_img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
          # red has hue 0 - 10 & 160 - 180 add another filter 
          # TODO  use Guassian mask
          RED_MIN1 = np.array([0, 100, 100],np.uint8)
          RED_MAX1 = np.array([10, 255, 255],np.uint8)        

          RED_MIN2 = np.array([160, 100, 100],np.uint8)
          RED_MAX2 = np.array([179, 255, 255],np.uint8)

          frame_threshed1 = cv2.inRange(hsv_img, RED_MIN1, RED_MAX1) 
          frame_threshed2 = cv2.inRange(hsv_img, RED_MIN2, RED_MAX2) 
          if cv2.countNonZero(frame_threshed1) + cv2.countNonZero(frame_threshed2) > 50:
            print ('Traffic Light Predicted = RED')
            return TrafficLight.RED

          YELLOW_MIN = np.array([40.0/360*255, 100, 100],np.uint8)
          YELLOW_MAX = np.array([66.0/360*255, 255, 255],np.uint8)
          frame_threshed3 = cv2.inRange(hsv_img, YELLOW_MIN, YELLOW_MAX)
          if cv2.countNonZero(frame_threshed3) > 50:
#            print ('Traffic Light Predicted = YELLOW')
#            return TrafficLight.YELLOW
####  Classifier predicting YELLOW for ground_truth GREEN, so swap until fixed
              print ('Traffic Light Predicted = GREEN')
              return TrafficLight.GREEN

          GREEN_MIN = np.array([90.0/360*255, 100, 100],np.uint8)
          GREEN_MAX = np.array([140.0/360*255, 255, 255],np.uint8)
          frame_threshed4 = cv2.inRange(hsv_img, GREEN_MIN, GREEN_MAX)
          if cv2.countNonZero(frame_threshed4) > 50:
#            print ('Traffic Light Predicted = GREEN')
#            return TrafficLight.GREEN
####  Classifier predicting GREEN for ground_truth YELLOW, so swap until fixed
              print ('Traffic Light Predicted = YELLOW')
              return TrafficLight.YELLOW
        else:
          desired_dim=(32,32)
          img_resized = cv2.resize(image, desired_dim, interpolation=cv2.INTER_LINEAR)
          img_ = np.expand_dims(np.array(img_resized), axis=0)

          predicted_state = self.model.predict_classes(img_)
          print ('Predicted_state = ', predicted_state)

#          predicted_state = self.test_an_image(image)
          if predicted_state == 'green':
            print ('Traffic Light Predicted = GREEN')
            return TrafficLight.GREEN
          elif predicted_state == 'yellow':
            print ('Traffic Light Predicted = YELLOW')
            return TrafficLight.YELLOW
          elif predicted_state == 'red':
            print ('Traffic Light Predicted = RED')
            return TrafficLight.RED

        if RED_SAFETY:
          return TrafficLight.RED
        else:
          return TrafficLight.UNKNOWN
