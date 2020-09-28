import tensorflow as tf
#import tensorflow.keras as keras
#from tensorflow.keras import datasets, layers, models
physical_devices = tf.config.list_physical_devices('GPU')
for dev in physical_devices:
    tf.config.experimental.set_memory_growth(dev, True)
from utils import square_func, sine_func, swish, cos_func, clippedsquare_func, NN_loss
import tensorflow_model_optimization as tfmot
import os
from kd_tree import build_tree
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
#import utils.py
from utils import MyCustomCallback, Accuracy, MaxAccuracyMult, AvgAccuracyMult, AccuracyMult, AccuracyDist, AccuracyNN

from base_model import Phi 
from combine_model import CombineModel, TempCallback

def call_learnable_index(config):
    def schedule(epoch):
        min_lr = 0.0005
        decay_factor = 2
        times_to_decay = math.log(config['lr']/min_lr)/math.log(decay_factor)
        decay_freq = config['EPOCHS']//times_to_decay
        lr = config['lr']/(decay_factor**(epoch//decay_freq))
        return lr


    callbacks = [tf.keras.callbacks.LearningRateScheduler(schedule)]

    if config['no_filters'] > 1:
        if config['learnable_depth'] != 0:
            epochs_pre_level = config['EPOCHS']//config['learnable_depth']
        else:
            epochs_pre_level = config['EPOCHS']
    init_temp = 1
    min_temp = 0.1
    decay = 2**((math.log(init_temp/min_temp)/math.log(2))/epochs_pre_level)
    callbacks.append(TempCallback(epochs_pre_level, decay))
    if config['no_filters']>1:
        my_model = get_model(config['depth'])
    else:
        my_model = Phi(config['out_dim'], config['in_dim'], config['filter_width1'], config['filter_width2'], config['phi_no_layers'], sine_func, False, config['degree'], config['batch_normalization'], 'base')

    if config['NN'] or config['NN_dist']:
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['lr'])#, clipnorm=1)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['lr'], clipnorm=4)

    metrics = []
    cifar = False
    if config['on_hot_encode']:
        my_model.compile(optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryAccuracy()])
    else:
        metrics = [Accuracy(config['accuracy_threshold'], name='accuracy'), Accuracy(config['accuracy_threshold']/10, name='accuracy_tenth'), Accuracy(config['accuracy_threshold']/100, name='accuracy_hundredth'), AccuracyMult(config['accuracy_mult_threshold'], name='rel_accuracy'), AvgAccuracyMult(name='avg_rel_accuracy'), MaxAccuracyMult(name='max_rel_accuracy')]
        if cifar:
            metrics.append(AccuracyDist(0.7, name='accuracy_dist_low'))
            metrics.append(AccuracyDist(0.3, name='accuracy_dist_mean'))
            metrics.append(AccuracyDist(0.5, name='accuracy_dist_high'))


    if config['NN'] or config['NN_dist']:
        if config['no_filters'] == 1:
            if config['NN_dist'] == False:
                my_model.compile(optimizer, loss=NN_loss, metrics=[AccuracyNN(0.1)])
            else:
                my_model.compile(optimizer, loss=tf.keras.losses.MeanSquaredError(), metrics=metrics)
            history = my_model.fit(queries, res, epochs=config['EPOCHS'], batch_size=config['train_size'], callbacks=callbacks, validation_data=(test_queries, test_res))
        else:
            hist=[]
            hist_indv=[]
            my_model.fit_partially_learnable(queries, res, test_queries, test_res, config['EPOCHS'], hist, hist_indv,tf.keras.losses.MeanSquaredError(), metrics, callbacks, config['lr'])
        #history = my_model.fit(queries, res, epochs=config['EPOCHS'], batch_size=config['train_size'], callbacks=callbacks, validation_data=(test_queries, test_res))
        #if config['no_filters'] != 1:
        #    hist_indv = []
        #    my_model.fit_base_only(queries, res, test_queries, test_res, config['EPOCHS'], hist_indv, metrics, optimizer)
    else:
        history = my_model.fit(queries, res, epochs=config['EPOCHS'], batch_size=config['train_size'], callbacks=callbacks, shuffle=False, validation_data=(queries, res))
    if config['no_filters'] != 1:
        for i, h in enumerate(hist):
            hist_df = pd.DataFrame(h.history) 
            with open(config['NAME']+str(i)+'indx_hist.json', 'w') as f:
                hist_df.to_json(f)

        for i, h in enumerate(hist_indv):
            hist_df = pd.DataFrame(h.history) 
            with open(config['NAME']+str(i)+'base_hist.json', 'w') as f:
                hist_df.to_json(f)
    else:
        hist_df = pd.DataFrame(history.history) 

        with open(config['NAME']+'_hist.json', 'w') as f:
            hist_df.to_json(f)
            #json.dump(hist, f)
    return my_model

def get_model(depth):
    if depth == 0:
        return Phi(config['out_dim'], config['in_dim'], config['filter_width1'], config['filter_width2'], config['phi_no_layers'], sine_func, False, config['degree'], config['batch_normalization'])


    base_models = []
    for i in range(config['no_filters']):
        base_models.append(get_model(depth-1))


    return CombineModel(config['in_dim'], config['out_dim'], base_models, config['on_hot_encode'], res, config['entropy'], depth<=config['learnable_depth'])

def call_tree(config, DB, queries, test_queries, res, test_res):
    processes = []
    min_dims = np.zeros(config['in_dim'])-0.5*config['MAX_VAL']
    max_dims = np.zeros(config['in_dim'])+0.5*config['MAX_VAL']
    if config['leaf_no_samples'] == 0:
        in_DB = None
        in_res = res
        in_test_res = test_res
    else:
        in_DB = DB
        in_res = None
        in_test_res = None

    my_model, _, processes = build_tree(config['depth'], config['no_filters'], config['in_dim'], queries, in_res, test_queries, in_test_res, processes, config['NAME'], "", 0, in_DB, config['k_th'], 0, config['leaf_no_samples'], min_dims, max_dims, config['NN_dist'], config['no_processes'])

    if len(processes) > 0:
        node = os.environ['NODE_NAME']
        with open("../../running_pids"+node+".txt", 'a') as f:
            for p in processes:
                f.write(str(p.pid)+"\n")
        for p in processes:
            p.wait()
    return my_model

def save_model_and_test(config, my_model):
    if config['no_filters'] == 1:
        if config['learnable_depth'] != 0:
            my_model.save_params(config['NAME']+'.m')
    else:
        cnt = my_model.get_params(config['NAME'], '', config['learnable_depth']!=0)
        with open(config['NAME']+'_tree.m', 'w') as f:
           f.write(cnt) 

    c_call = "../../cpp_model_parallel "+str(config['no_filters'])+" " + str(config['in_dim'])+" " + str(config['out_dim']) +' ' + str(config['depth'])+ ' ' + 'test' + ' ' + config['NAME'] + ' -1' + ' ' + str(config['degree'])
    print(c_call)
    os.system(c_call)

