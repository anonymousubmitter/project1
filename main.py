import time
import bisect
import sys
import datetime
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

from model_utils import call_learnable_index, call_tree, save_model_and_test, learn_index_from_res_val
from data_utils import get_data


read_conf = int(sys.argv[2]) == 1
if read_conf:
    f = open(sys.argv[1],) 
    config = json.load(f) 
else:
    config={}
    #config['compile'] = compile_model
    config['NAME'] = sys.argv[1]
    config['NN_dist'] = True
    config['NN'] = False
    config['RangeSum'] = False
    config['KV'] = False
    config['EPOCHS'] = 20
    config['train_size'] = 50000
    config['test_size'] = 10000
    config['learnable_depth'] = 1
    config['depth'] = 1
    config['no_comp'] = 1000
    config['no_filters'] = 2#config['n']//300
    config['n'] = 500*(config['no_filters']**config['depth'])
    config['k_th'] = 1
    if config['KV']:
        config['train_size'] = config['n']
        config['degree'] = 128
        config['lr']=0.001
        config['batch_normalization'] = False
        config['filter_width1'] = 25
        config['filter_width2'] =10
        config['phi_no_layers']=1
        config['accuracy_threshold'] = 1
        config['in_dim'] = 1
    else:
        config['in_dim'] = 2
        config['lr']=0.01
        config['degree'] = 1
        config['batch_normalization'] = False
        config['filter_width1'] = 60
        config['filter_width2'] = 30 
        config['phi_no_layers'] = 4
        config['accuracy_threshold'] = 0.1
    
    if config['NN']:
        config['out_dim'] = config['in_dim']
    else:
        config['out_dim'] = 1
    config['accuracy_mult_threshold'] = 0.1
    config['batch_size'] = config['n']#//10
    config['on_hot_encode'] = False
    config['entropy'] = True
    config['leaf_no_samples'] = 100000
    config['data_loc'] = ""
    config['MAX_VAL']=10

with open('conf.json', 'w') as f:
    json.dump(config, f)
if "only_begin" in config:
    only_begin = config["only_begin"]
else:
    only_begin = False
DB, queries, test_queries, res, test_res = get_data(config['n'], config['in_dim'], config['out_dim'], config['NN'], config['NN_dist'], config['RangeSum'], config['train_size'], config['test_size'], config['on_hot_encode'], config['k_th'], config['no_comp'], config['data_loc'], config['leaf_no_samples'] == 0, config['q_range'], config['agg_type'], config['MAX_VAL'], config['db_sel_col'], config['db_ag_col'], config['query_loc'], config['NON_NULL'], config["train_data_size"], only_begin, config["pred_dim"], config["pred_dim_pairs"], config["with_angle"])

if  config['data_loc'][0] != "/":
    np.savetxt('DB.txt', DB, delimiter=',', fmt='%.16f');
np.save('dist_res', res);
#np.savetxt('train_queries.txt', queries, delimiter=',', fmt='%.16f');
#np.savetxt('train_res.txt', res, delimiter=',', fmt='%.16f');
if config['leaf_no_samples'] == 0:
    np.savetxt('test_queries.txt', test_queries, delimiter=',', fmt='%.16f');
    np.savetxt('test_res.txt', test_res, delimiter=',', fmt='%.16f');

start = time.time()

if 'res_val_partition' in config and config['res_val_partition']:
    my_model = learn_index_from_res_val(config, DB, queries, test_queries, res, test_res)
else:
    if config['learnable_depth'] != 0:
        my_model = call_learnable_index(config)
    else:
        my_model = call_tree(config, DB, queries, test_queries, res, test_res)

end = time.time()
print("TOOK:"+str(end-start))

save_model_and_test(config, my_model)



