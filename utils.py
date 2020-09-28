import sys
import time
import bisect
import datetime

import tensorflow as tf

import numpy as np
import pandas as pd

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from sklearn.neighbors import NearestNeighbors

from tensorflow_privacy.privacy.analysis import privacy_ledger
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp_from_ledger
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers import dp_optimizer

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer


import json

from tensorflow.keras import backend as K
import tensorflow.keras as keras

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

class SavePred(tf.keras.callbacks.Callback):
    def __init__(self, test):
        self.test = test
    
    def on_epoch_end(self, epoch, logs={}):
        if  epoch%5 == 0:
            preds=model.predict_on_batch(self.test)
            np.savetxt('preds'+str(epoch)+'.txt', preds, delimiter=',', fmt='%.16f');

class PrintEpochNo(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if  epoch%1 == 0:
            tf.print("Epoch no " + str(epoch)+ " loss " + str(logs.get('loss'))  + " val_loss " + str(logs.get('val_loss')) + " avg_acc " + str(logs.get('avg_rel_accuracy'))+ " val_avg_acc " + str(logs.get('val_avg_rel_accuracy')), output_stream=sys.stdout)

class AccuracyNN(tf.keras.metrics.Metric):
    def __init__(self, threshold, name='accuracy_NN', **kwargs):
        super(AccuracyNN, self).__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name='ac', initializer='zeros', dtype=tf.float64)
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, tf.float64)
        y_true = tf.cast(y_true, tf.float64)
        
        dist = tf.math.sqrt(tf.reduce_sum(tf.math.square(y_true-y_pred), axis=-1) + 1e-8)

        self.accuracy.assign_add(tf.reduce_mean(tf.cast(dist<=self.threshold, tf.float64)))

    def result(self):
      return self.accuracy

    def reset_states(self):
      # The state of the metric will be reset at the start of each epoch.
      self.accuracy.assign(0.)

class AccuracyDist(tf.keras.metrics.Metric):
    def __init__(self, threshold, name='accuracy', **kwargs):
        super(AccuracyDist, self).__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name='ac', initializer='zeros', dtype=tf.float64)
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, tf.float64)
        y_true = tf.cast(y_true, tf.float64)
        
        y_pred = y_pred>self.threshold
        y_true = y_true>self.threshold

        self.accuracy.assign_add(tf.reduce_mean(tf.cast(y_pred==y_true, tf.float64)))

    def result(self):
      return self.accuracy

    def reset_states(self):
      # The state of the metric will be reset at the start of each epoch.
      self.accuracy.assign(0.)

class MaxAccuracyMult(tf.keras.metrics.Metric):
    def __init__(self, name='accuracy', **kwargs):
        super(MaxAccuracyMult, self).__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name='ac', initializer='zeros', dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, tf.float64)
        y_true = tf.cast(y_true, tf.float64)

        err = tf.math.abs(y_pred-y_true)/y_true
        self.accuracy.assign_add(tf.reduce_max(err))

    def result(self):
      return self.accuracy

    def reset_states(self):
      self.accuracy.assign(0.)

class AvgAccuracyMult(tf.keras.metrics.Metric):
    def __init__(self, test_queries, train_queries, name='accuracy', **kwargs):
        super(AvgAccuracyMult, self).__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name='ac', initializer='zeros', dtype=tf.float64)
        self.count = self.add_weight(name='count', initializer='zeros', dtype=tf.float64)
        #self.train_queries = train_queries
        #self.test_queries = test_queries

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, tf.float64)
        y_true = tf.cast(y_true, tf.float64)

        err = tf.math.abs(y_pred-y_true)/y_true
        self.accuracy.assign_add(tf.reduce_mean(err))
        self.count.assign_add(1.0)

    def result(self):
      return self.accuracy/self.count

    def reset_states(self):
      self.count.assign(0.)
      self.accuracy.assign(0.)

class SMSE(tf.keras.metrics.Metric):
    def __init__(self, name='smse_accuracy', **kwargs):
        super(SMSE, self).__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name='ac', initializer='zeros', dtype=tf.float64)
        self.count = self.add_weight(name='count', initializer='zeros', dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, tf.float64)
        y_true = tf.cast(y_true, tf.float64)

        err = tf.math.abs(y_pred-y_true)

        self.accuracy.assign_add(tf.reduce_mean(err))
        self.count.assign_add(1)

    def result(self):
      return self.accuracy/self.count

    def reset_states(self):
      self.count.assign(0.)
      self.accuracy.assign(0.)

class AccuracyMult(tf.keras.metrics.Metric):
    def __init__(self, threshold, name='accuracy', **kwargs):
        super(AccuracyMult, self).__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name='ac', initializer='zeros', dtype=tf.float64)
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, tf.float64)
        y_true = tf.cast(y_true, tf.float64)

        eps = tf.zeros_like(y_pred)+self.threshold
        err = tf.math.abs(y_pred-y_true)/y_true

        true = tf.math.less(err, eps)
        self.accuracy.assign_add(tf.reduce_mean(tf.cast(true, tf.float64)))

    def result(self):
      return self.accuracy

    def reset_states(self):
      self.accuracy.assign(0.)

class Accuracy(tf.keras.metrics.Metric):
    def __init__(self, threshold, name='accuracy', **kwargs):
        super(Accuracy, self).__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name='ac', initializer='zeros', dtype=tf.float64)
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, tf.float64)
        y_true = tf.cast(y_true, tf.float64)

        eps = tf.zeros_like(y_pred)+self.threshold
        false = tf.math.greater(tf.math.abs(y_pred-y_true), eps)
        self.accuracy.assign_add(1-tf.reduce_mean(tf.cast(false, tf.float64)))

    def result(self):
      return self.accuracy

    def reset_states(self):
      self.accuracy.assign(0.)

def mse(y_true, y_pred):
    return  K.mean((K.sum(K.square(y_true-y_pred), axis=-1)))

def NN_loss(y_true, y_pred):
    return  K.mean(K.sqrt(K.abs(K.sum(K.square(y_true-y_pred), axis=-1)+1e-8)))

def exp_func(x):
    in_exp = tf.math.multiply(tf.math.square(x), -1)
    exp = tf.math.multiply((1-K.square(x)), tf.math.exp(in_exp))
    return exp
def cos_func(x):
    return K.cos(x)
def clippedsquare_func(x):
    return K.clip(K.square(x), -1, 1)+x
def square_func(x):
    return K.square(x)
def sine_func(x):
    return K.sin(x)
def swish(x, beta=1.0):
    return x * K.sigmoid(beta * x)

def compute_epsilon(n, steps, noise_multiplier, batch_size, delta):
  """Computes epsilon value for given hyperparameters."""
  if noise_multiplier == 0.0:
    return float('inf')
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  sampling_probability = batch_size / n
  rdp = compute_rdp(q=sampling_probability,
                    noise_multiplier=noise_multiplier,
                    steps=steps,
                    orders=orders)
  return get_privacy_spent(orders, rdp, target_delta=delta)[0]

class MyCustomCallback(tf.keras.callbacks.Callback):
  def __init__(self, batch_size, n, train_size, noise_multiplier, delta):
    self.n = n
    self.train_size = train_size
    self.batch_size = batch_size
    self.noise_multiplier = noise_multiplier
    self.delta = delta

  def on_epoch_end(self, epoch, logs=None):
    print(' eps {}'.format(compute_epsilon(self.n, epoch*self.train_size//self.batch_size, self.noise_multiplier, self.batch_size, self.delta)))


