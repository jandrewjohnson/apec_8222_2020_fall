import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def run_cnn():
   mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
   learning_rate = 0.0001
   epochs = 10
   batch_size = 50