import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, SpatialDropout2D
from functools import partial

class LabelPredictor_mnist(Model):
    def __init__(self, num_class, use_activation=False):
        super(LabelPredictor_mnist, self).__init__()
        self.d1 = Dense(128, activation='relu')
        if use_activation :
          self.d2 = Dense(num_class, activation='softmax')
        else :
          self.d2 = Dense(num_class)

    def __call__(self, feats, is_train):  
        feats = self.d1(feats)
        return self.d2(feats)


class FeatureGenerator_mnist(Model):
  def __init__(self):
    super(FeatureGenerator_mnist, self).__init__() 
    self.normalise = lambda x: (tf.cast(x, tf.float32) - tf.reduce_mean(tf.cast(x, tf.float32))) / 255.0
    self.conv1 = Conv2D(64, 5, activation='relu')
    self.conv2 = Conv2D(128, 5, activation='relu')
    self.maxpool = MaxPool2D(2)
    self.flatten = Flatten()

    
  def call(self, x, is_train):
    x = self.normalise(x)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.maxpool(x)
    x = self.flatten(x)

    return x





  
def get_label_predictor(dataset):
    if dataset == 'mnist':
      return partial(LabelPredictor_mnist, num_class=10)

def get_feature_generator(dataset):
    if dataset == 'mnist':
      return FeatureGenerator_mnist
