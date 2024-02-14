import tensorflow as tf
import numpy as np
import pickle as pkl
from tensorflow.keras.datasets import mnist
from functools import partial
import os
import cv2
from scipy.io import loadmat

def load_data_iterator(dataset_name):
    if dataset_name == 'mnist':
        return load_mnist_dataset

def load_mnist_dataset(dataset_name_source, dataset_name_target, num_sample_per_class
    , batch_size):
    (x_train_A, y_train_A), (x_test_A, y_test_A) = mnist.load_data()
    mnist_train = (x_train_A > 0).reshape(60000, 28, 28, 1).astype(np.uint8) * 255
    y_train_A = y_train_A.astype(np.int32)
    y_test_A = y_test_A.astype(np.int32)
    mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
    mnist_test = (x_test_A > 0).reshape(10000, 28, 28, 1).astype(np.uint8) * 255
    mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
    mnistm = pkl.load(open('mnistm_data.pkl', 'rb'))
    mnistm_train = mnistm['train']
    mnistm_test = mnistm['test']
    mnistm_valid = mnistm['valid']
    train_ds = tf.data.Dataset.from_tensor_slices((mnist_train, y_train_A)).shuffle(60000).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((mnist_test, y_test_A)).shuffle(60000).batch(batch_size)
    train_ds_B = tf.data.Dataset.from_tensor_slices((mnistm_train, y_train_A)).batch(batch_size)
    train_A_buffer_ids=np.concatenate([np.where(y_train_A==i)[0][:num_sample_per_class] for i in range(10)])
    # print(np.shape(train_A_buffer_ids))
    train_ds_A2 = tf.data.Dataset.from_tensor_slices((mnist_train[train_A_buffer_ids], y_train_A[train_A_buffer_ids])).shuffle(60000).repeat().batch(batch_size)
    test_ds_B = tf.data.Dataset.from_tensor_slices((mnistm_test, y_test_A)).batch(batch_size)
    return train_ds, test_ds, train_ds_B, test_ds_B, train_ds_A2

  

  
if __name__ == '__main__':

 #   imagepath_source, imagepath_target_train, imagepath_target_test, labels_source, labels_target_train, labels_target_test = load_office('amazon', 'dslr')
 #   print(imagepath_source[:10])
 #   print(imagepath_target_train[:10])
 #   print(imagepath_target_test[:10])
    # source_dataset_train, target_dataset_train, target_dataset_test, mem_dataset = create_data(imagepath_source, imagepath_target_train
    #     , imagepath_target_test, labels_source, labels_target_train, labels_target_test, 10
    # , 32)
    source_dataset_train, source_dataset_test, target_dataset_train, target_dataset_test, mem_dataset = load_mnist_dataset('', '', 16, 32)
    for source_data, source_label in source_dataset_test:
        print(source_data)
        print(source_label)
        break
    for data, label in target_dataset_test:
        print(data)
        print(label)
        break