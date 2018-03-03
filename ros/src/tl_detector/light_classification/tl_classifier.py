from styx_msgs.msg import TrafficLight
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import load_model

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        test(test_file, model=load_model('./model.h5'))

    def test_an_image(file_path, model):
       """
       resize the input image to [32, 32, 3], then feed it into the NN for prediction
       :param file_path:
       :return:
       """

       desired_dim=(32,32)
       img = cv2.imread(file_path)
       img_resized = cv2.resize(img, desired_dim, interpolation=cv2.INTER_LINEAR)
       img_ = np.expand_dims(np.array(img_resized), axis=0)

       predicted_state = model.predict_classes(img_)

       return predicted_state



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
        predicted_state = model.predict_classes(img_)

        return predicted_state
#        return TrafficLight.UNKNOWN
