import tensorflow as tf
import gc
from utils import swish, sine_func, AccuracyMult, AvgAccuracyMult, MaxAccuracyMult, NN_loss, SMSE, mse, PrintEpochNo, n_mse, MAE, Smooth_RMAE, SaveBestCallBack, Smooth_dist_RMAE_NN, Smooth_RMAE_NN
from base_model_dim import PhiDim 
from base_model_degree import PhiDegree
import json
import sys
import math
from tensorflow.keras import backend as K
import os
import numpy as np
import pandas as pd
physical_devices = tf.config.list_physical_devices('GPU')
for dev in physical_devices:
    tf.config.experimental.set_memory_growth(dev, True)
K.set_floatx('float64')


f = open('conf.json') 
config = json.load(f) 
no = int(sys.argv[1])
base_name = sys.argv[2]
path = sys.argv[3]

if config["out_dim"] == 1 and config["NN"] == True:
    if config['data_loc'][0] == "/":
        db = np.load(config['data_loc'])
    else:
        db = np.loadtxt('DB.txt', delimiter=',')
else:
    db = None
#train = np.loadtxt('test'+str(no)+'_queries.txt', delimiter=',')
#res = np.loadtxt('test'+str(no)+'_res.txt', delimiter=',')
test = np.load('test'+str(no)+'_queries.npy')
test_res = np.load('test'+str(no)+'_res.npy').astype(float)
train = np.load('queries'+str(no)+'.npy')
res = np.load('res'+str(no)+'.npy').astype(float)
tf.print(test.shape, output_stream=sys.stdout)
tf.print(test_res.shape, output_stream=sys.stdout)
tf.print(train.shape, output_stream=sys.stdout)
tf.print(res.shape, output_stream=sys.stdout)

one_model_per_outdim = config['ONE_MODEL_PER_DIM']
if config['out_dim'] == 1 or (not one_model_per_outdim):
    if config['degree'] == 1:
        if config["LOAD_MODEL"] != "-1":
            from base_conv_model import Phi 
            model = Phi(config['out_dim'], config['in_dim'], config['filter_width1'], config['filter_width2'], config['phi_no_layers'], sine_func, False, config['degree'], config['batch_normalization'], 1, True, config["LOAD_MODEL"])
        else:
            from base_model import Phi 
            if config["1_act_f"] == "sine":
                act_f_1 = sine_func
            else:
                act_f_1 = swish

            if config["2_act_f"] == "sine":
                act_f_2 = sine_func
            elif config["2_act_f"] == "swish":
                act_f_2 = swish
            else:
                act_f_2 = tf.nn.relu
            if no == -1:
                model = Phi(config['no_leaves'], config['in_dim'], config['filter_width1'], config['filter_width2'], config['phi_no_layers'], act_f_1, True, config['degree'], config['batch_normalization'], act_f_2)
            else:
                model = Phi(config['out_dim'], config['in_dim'], config['filter_width1'], config['filter_width2'], config['phi_no_layers'], act_f_1, False, config['degree'], config['batch_normalization'], act_f_2)
    else:
        model = PhiDegree(config['out_dim'], config['in_dim'], config['filter_width1'], config['filter_width2'], config['phi_no_layers'], sine_func, False, config['degree'], config['batch_normalization'])
else:
    model = PhiDim(config['out_dim'], config['in_dim'], config['filter_width1'], config['filter_width2'], config['phi_no_layers'], sine_func, False, config['degree'], config['batch_normalization'])


base_lr = config['lr']
min_lr = config['min_lr']
decay_factor = 3
times_to_decay = math.log(base_lr/min_lr)/math.log(decay_factor)
decay_freq = config['EPOCHS']//times_to_decay
def schedule(epoch):
    lr = base_lr/(decay_factor**(epoch//decay_freq))
    return lr

callbacks = [tf.keras.callbacks.LearningRateScheduler(schedule), PrintEpochNo()]
#if no != -1:
    #callbacks.append(SaveBestCallBack(test, test_res, model))
#    callbacks.append(SaveBestCallBack(train, res, model, str(no)))
#if no == -2:
#    callbacks.append(SaveBestCallBack(train, res, model, "train"))
if no == -1:
    metrics = [tf.keras.metrics.CategoricalAccuracy()]
else:
    if not config['NN']:
        #metrics = [AccuracyMult(config['accuracy_mult_threshold'], name='rel_accuracy'), AvgAccuracyMult(None, None, name='avg_rel_accuracy'), MaxAccuracyMult(name='max_rel_accuracy'), SMSE(), MAE()]
        metrics = [SMSE(), MAE(), Smooth_RMAE(1e-6)]
    else:
        metrics = [Smooth_RMAE_NN(1e-6, test, db), Smooth_dist_RMAE_NN(1e-6, test, train, db)]
        callbacks.append(SaveBestCallBack(test, test_res, model, "mse"+str(no)))
        callbacks.append(SaveBestCallBack(test, test_res, model, "rel_acc"+str(no)))
optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)
if no == -1:
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
else:
    if config['normalized_loss']:
        loss = n_mse
    else:
        loss = mse

'''
def generator3():
    queries_t = tf.random.shuffle(train)
    dim = config['in_dim']//2
    queries_t = tf.reshape(queries_t, (-1, dim, 2))

    DB_t = tf.random.shuffle(db)
    batch_size=db.shape[0]//config['batch_size']
    no_batches = config['batch_size']
    start = 0
    train_curr = []
    label = []
    for i in range(no_batches):
        for j in range(i*batch_size, (i+1)*batch_size):
            res = tf.transpose(tf.reduce_all([tf.logical_and(DB_t[j, d]>=queries_t[:, d, 0], DB_t[j, d]<queries_t[i, d, 1]) for d in range(dim)], axis=0))
            
            train_curr.append(tf.reshape(queries_t, (-1, dim*2)))
            label.append(res)

        train_curr = tf.concat(train_curr, axis=0)

        label = tf.cast(tf.concat(label, axis=0), dtype=tf.float64)*DB_t.shape[0]
        yield train_curr, tf.reshape(label, (-1, config['out_dim']))
        train_curr = []
        label = []
    return
'''


#dataset = tf.data.Dataset.from_generator(generator3, (tf.float32, tf.float32), output_shapes=((None, config['in_dim']), (None, config['out_dim'])))

if config["LOAD_MODEL"] == "-1":
    model.compile(optimizer, loss=loss, metrics=metrics)
    print(test_res)
    h = model.fit(train, res, epochs=config['EPOCHS'], batch_size=train.shape[0]//config['batch_size'], callbacks=callbacks, validation_data=(test, test_res), validation_steps=1, verbose=0, shuffle=False)
    #h = model.fit(dataset, epochs=config['EPOCHS'], callbacks=callbacks, validation_data=(train, res), validation_steps=1, verbose=0)
    model.summary()
else:
    enc_train=[]
    enc_test=[]
    for i in range(config['batch_size']):
        enc_train.append(model.get_mn_enc(train[i*(train.shape[0]//config['batch_size']):(i+1)*(train.shape[0]//config['batch_size'])]))
        enc_test.append(model.get_mn_enc(test[i*(test.shape[0]//config['batch_size']):(i+1)*(test.shape[0]//config['batch_size'])]))
    np.savetxt('enc_train.txt',np.concatenate(enc_train, axis=0), delimiter=',', fmt='%.16f');
    np.savetxt('enc_test.txt', np.concatenate(enc_test, axis=0), delimiter=',', fmt='%.16f');
    db_enc =model.get_mn_enc(db) 
    np.savetxt('enc_db.txt', np.concatenate(db_enc, axis=0), delimiter=',', fmt='%.16f');

if config["SAVE_PREDS"]:
    preds=model.predict_on_batch(test)
    np.savetxt('preds.txt', preds, delimiter='\t', fmt='%.16f');

hist_df = pd.DataFrame(h.history) 
with open(base_name+str(no)+'base_hist.json', 'w') as f:
    hist_df.to_json(f)

if (config['out_dim']!=1 and one_model_per_outdim) or config['degree']!=1:
    if config['degree']!=1:
        model.save_params(base_name+path+"0")
    else:
        model.save_params(base_name+path)
else:
    model.save_params(base_name+path+"00.m")

c_call = "/tank/users/zeighami/project1/cpp_model_parallel 1 " + str(config['in_dim'])+" " + str(config['out_dim']) +' 0 ' + 'test'+str(no)+  ' ' + base_name+path + ' ' + str(no) + " " + str(config['degree'])
print(c_call)
os.system(c_call)


#os.system('rm queries'+str(no)+'.txt')
#os.system('rm res'+str(no)+'.txt')

#os.system('rm test'+str(no)+'_queries.txt')
#os.system('rm test'+str(no)+'_res.txt')

