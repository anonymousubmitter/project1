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

class SaveBestCallBack(tf.keras.callbacks.Callback):
    def __init__(self, test, test_res, model, ext="", **kwargs):
        self.test = test
        self.test_res = test_res
        self.min_err = 1000000
        self.model = model
        self.ext = ext
        np.savetxt("test_qs"+ext+".txt", test)
        np.savetxt("test_qs_res"+ext+".txt", test_res)

    def on_epoch_end(self, epoch, logs=None):
        y_pred = tf.cast(self.model.call(self.test), tf.float64)
        y_true = tf.cast(self.test_res, tf.float64)
        if self.ext[:3] == "mse":
            err = tf.reduce_mean(tf.sqrt(tf.reduce_sum((y_true-y_pred)**2, axis=1)))
        else:
            dist_pred = tf.sqrt(tf.reduce_sum((y_pred-self.test)**2, axis=1))
            dist_true = tf.sqrt(tf.reduce_sum((y_true-self.test)**2, axis=1))
            err = tf.reduce_mean(tf.abs(dist_true-dist_pred)/(tf.math.maximum(dist_true, 1e-6)))

        if err < self.min_err:
            np.savetxt("best_pred"+self.ext+".txt", y_pred)
            self.min_err = err

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
            #tf.print("asdfasdfasdf", output_stream=sys.stdout)
            #tf.print("Epoch no " + str(epoch)+ " loss " + str(logs.get('loss'))  + " val_loss " + str(logs.get('val_loss')) + " avg_acc " + str(logs.get('avg_rel_accuracy'))+ " val_avg_acc " + str(logs.get('val_avg_rel_accuracy'))+ " mae " + str(logs.get('mae'))+ " val_mae " + str(logs.get('val_mae'))+ " smooth_rmae " + str(logs.get('smooth_rmae'))+ " val_smooth_rmae " + str(logs.get('val_smooth_rmae'))+ " acc " + str(logs.get('categorical_accuracy'))+ " val_acc " + str(logs.get('val_categorical_accuracy')))+ " NN_dist_acc " + str(logs.get('smooth_dist_rmae_NN'))+ " val_dist_NN_acc " + str(logs.get('val_smooth_dist_rmae_NN')+ " NN_acc " + str(logs.get('smooth_rmae_NN'))+ " val_NN_acc " + str(logs.get('val_smooth_rmae_NN')), output_stream=sys.stdout)
            tf.print("Epoch no " + str(epoch)+ " loss " + str(logs.get('loss'))  + " val_loss " + str(logs.get('val_loss')) + " avg_acc " + str(logs.get('avg_rel_accuracy'))+ " val_avg_acc " + str(logs.get('val_avg_rel_accuracy'))+ " mae " + str(logs.get('mae'))+ " val_mae " + str(logs.get('val_mae'))+ " smooth_rmae " + str(logs.get('smooth_rmae'))+ " val_smooth_rmae " + str(logs.get('val_smooth_rmae'))+ " acc " + str(logs.get('categorical_accuracy'))+ " val_acc " + str(logs.get('val_categorical_accuracy'))+" dist_NN_acc " + str(logs.get('smooth_dist_rmae_NN'))+" val_dist_NN_acc " + str(logs.get('val_smooth_dist_rmae_NN'))+" NN_acc " + str(logs.get('smooth_rmae_NN'))+" val_NN_acc " + str(logs.get('val_smooth_rmae_NN')), output_stream=sys.stdout)
            #tf.print(" NN_dist_acc " + str(logs.get('smooth_dist_rmae_NN')) , output_stream=sys.stdout)
            #tf.print(" val_dist_NN_acc " + str(logs.get('val_smooth_dist_rmae_NN')), output_stream=sys.stdout)
            #tf.print(" NN_acc " + str(logs.get('smooth_rmae_NN')), output_stream=sys.stdout)
            #tf.print(" val_NN_acc " + str(logs.get('val_smooth_rmae_NN')), output_stream=sys.stdout)


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

        err = tf.math.abs(y_pred-y_true)/tf.math.abs(y_true)
        self.accuracy.assign_add(tf.reduce_mean(err))
        self.count.assign_add(1.0)

    def result(self):
      return self.accuracy/self.count

    def reset_states(self):
      self.count.assign(0.)
      self.accuracy.assign(0.)

class Smooth_dist_RMAE_NN(tf.keras.metrics.Metric):
    def __init__(self, smooth, test_queries, queries, db=None, name='smooth_dist_rmae_NN', **kwargs):
        super(Smooth_dist_RMAE_NN, self).__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name='ac', initializer='zeros', dtype=tf.float64)
        self.count = self.add_weight(name='count', initializer='zeros', dtype=tf.float64)
        #self.val = val
        self.smooth = smooth
        self.queries = tf.convert_to_tensor(queries)
        self.test_queries = tf.convert_to_tensor(test_queries)
        if db is not None:
            self.db = tf.convert_to_tensor(db)
        else:
            self.db = None


    def update_state(self, y_true, y_pred, sample_weight=None):
        self.accuracy.assign_add(0.)
        self.count.assign_add(1)
        return
        if tf.shape(y_true)[0] != tf.shape(self.test_queries)[0]:
            self.accuracy.assign_add(0.)
            self.count.assign_add(1)
        else:
            queries = self.test_queries
            #queries = self.queries
            if self.db is not None:
                y_pred = tf.cast(y_pred, tf.int64)
                y_true = tf.cast(y_true, tf.int64)
                y_pred = tf.gather(self.db, tf.reshape(y_pred, [-1]))
                y_true = tf.gather(self.db, tf.reshape(y_true, [-1]))


            y_pred = tf.cast(y_pred, tf.float64)
            y_true = tf.cast(y_true, tf.float64)

            #no_samples = 0
            #if tf.shape(y_true)[0] < tf.shape(queries)[0]:
            #    no_samples = tf.shape(y_true)[0]
            #else:
            #    no_samples = tf.shape(queries)[0]

            #tf.print("HERERE", output_stream=sys.stdout)
            #tf.print(no_samples, output_stream=sys.stdout)
            #tf.print(tf.cast(self.count, tf.int32), output_stream=sys.stdout)

            tf.print("HERERE", output_stream=sys.stdout)
            tf.print(tf.shape(y_true)[0], output_stream=sys.stdout)
            tf.print(tf.shape(y_pred)[0], output_stream=sys.stdout)
            tf.print(self.test_queries.shape[0], output_stream=sys.stdout)
            dist_pred = tf.sqrt(tf.reduce_sum((y_pred-queries)**2, axis=1))
            dist_true = tf.sqrt(tf.reduce_sum((y_true-queries)**2, axis=1))
            #dist_pred = tf.sqrt(tf.reduce_sum((y_pred[:no_samples]-queries[tf.cast(self.count, tf.int32):tf.cast(self.count, tf.int32)+no_samples])**2, axis=1))
            #dist_true = tf.sqrt(tf.reduce_sum((y_true[:no_samples]-queries[tf.cast(self.count, tf.int32):tf.cast(self.count, tf.int32)+no_samples])**2, axis=1))
            err = tf.abs(dist_true-dist_pred)/(tf.math.maximum(dist_true, self.smooth))

            #tf.print("EHRERE", output_stream=sys.stdout)
            #tf.print(err, output_stream=sys.stdout)
            self.accuracy.assign_add(tf.reduce_mean(err))
            self.count.assign_add(1)
    def result(self):
      return self.accuracy/self.count

    def reset_states(self):
      self.count.assign(0.)
      self.accuracy.assign(0.)

class Smooth_RMAE_NN(tf.keras.metrics.Metric):
    def __init__(self, smooth, queries, db=None, name='smooth_rmae_NN', **kwargs):
        super(Smooth_RMAE_NN, self).__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name='ac', initializer='zeros', dtype=tf.float64)
        self.count = self.add_weight(name='count', initializer='zeros', dtype=tf.float64)
        #self.val = val
        self.smooth = smooth
        self.queries = queries
        if db is not None:
            self.db = tf.convert_to_tensor(db)
        else:
            self.db = None


    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.db is not None:
            y_pred = tf.cast(y_pred, tf.int64)
            y_true = tf.cast(y_true, tf.int64)
            y_pred = tf.gather(self.db, tf.reshape(y_pred, [-1]))
            y_true = tf.gather(self.db, tf.reshape(y_true, [-1]))


        y_pred = tf.cast(y_pred, tf.float64)
        y_true = tf.cast(y_true, tf.float64)

        #dist_pred = tf.sqrt(tf.reduce_sum((y_pred-self.queries)**2, axis=1))
        #dist_true = tf.sqrt(tf.reduce_sum((y_true[:500]-self.queries[:500, :y_true.shape[1]])**2, axis=1))
        dist_err = tf.sqrt(tf.reduce_sum((y_true-y_pred)**2, axis=1))
        #tf.print(dist_true, output_stream=sys.stdout)
        #tf.print(dist_err, output_stream=sys.stdout)

        #val_size=int(tf.shape(y_true)[0]/2)

        #if self.val:
        #err = tf.math.abs(y_pred[:val_size]-y_true[:val_size])/(tf.math.maximum(y_true[:val_size], self.smooth))
        #err = dist_err/(tf.math.maximum(dist_true, self.smooth))
        err = dist_err#/(tf.reduce_mean(dist_true))
        #else:
        #    err = tf.math.abs(y_pred[val_size:]-y_true[val_size:])/(tf.math.maximum(y_true[val_size:], self.smooth))

        self.accuracy.assign_add(tf.reduce_mean(err))
        self.count.assign_add(1)

    def result(self):
      return self.accuracy/self.count

    def reset_states(self):
      self.count.assign(0.)
      self.accuracy.assign(0.)

class Smooth_RMAE(tf.keras.metrics.Metric):
    def __init__(self, smooth, name='smooth_rmae', **kwargs):
        super(Smooth_RMAE, self).__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name='ac', initializer='zeros', dtype=tf.float64)
        self.count = self.add_weight(name='count', initializer='zeros', dtype=tf.float64)
        #self.val = val
        self.smooth = smooth

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(tf.clip_by_value(y_pred, clip_value_min=0, clip_value_max=tf.float32.max), tf.float64)
        y_true = tf.cast(y_true, tf.float64)

        #val_size=int(tf.shape(y_true)[0]/2)

        #if self.val:
        #err = tf.math.abs(y_pred[:val_size]-y_true[:val_size])/(tf.math.maximum(y_true[:val_size], self.smooth))
        err = tf.math.abs(y_pred-y_true)/(tf.math.maximum(tf.math.abs(y_true), self.smooth))
        #else:
        #    err = tf.math.abs(y_pred[val_size:]-y_true[val_size:])/(tf.math.maximum(y_true[val_size:], self.smooth))

        self.accuracy.assign_add(tf.reduce_mean(err))
        self.count.assign_add(1)

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

class MAE(tf.keras.metrics.Metric):
    def __init__(self, name='mae', **kwargs):
        super(MAE, self).__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name='ac', initializer='zeros', dtype=tf.float64)
        self.count = self.add_weight(name='ac', initializer='zeros', dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, tf.float64)
        y_true = tf.cast(y_true, tf.float64)

        err = tf.math.abs(y_pred-y_true)

        self.accuracy.assign_add(tf.reduce_mean(tf.cast(err, tf.float64)))
        self.count.assign_add(1)

    def result(self):
      return self.accuracy/self.count

    def reset_states(self):
      self.accuracy.assign(0.)
      self.count.assign(0.)
class Normed_MAE(tf.keras.metrics.Metric):
    def __init__(self, name='normed_mae', **kwargs):
        super(Normed_MAE, self).__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name='ac', initializer='zeros', dtype=tf.float64)
        self.count = self.add_weight(name='ac', initializer='zeros', dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, tf.float64)
        y_true = tf.cast(y_true, tf.float64)

        #err = tf.math.abs(y_pred-y_true)/tf.reduce_mean(tf.math.abs(y_true))
        err = tf.math.abs(y_pred-y_true)/833.4949004147102

        self.accuracy.assign_add(tf.reduce_mean(tf.cast(err, tf.float64)))
        self.count.assign_add(1)

    def result(self):
      return self.accuracy/self.count

    def reset_states(self):
      self.accuracy.assign(0.)
      self.count.assign(0.)

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

def n_mse(y_true, y_pred):
    return  K.mean((K.sum(K.square(y_true-y_pred), axis=-1))/(K.sum(K.square(y_true), axis=-1)))

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


