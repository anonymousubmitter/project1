import random
import subprocess
import json
import os

config={}
config['NN_dist'] = True
config['NN'] = False
config['RangeSum'] = False
config['KV'] = False
if config['NN_dist']:
    config['NAME'] = "test_NN_dist"
elif config['NN']:
    config['NAME'] = "test_NN"
elif config['RangeSum']:
    config['NAME'] = "test_RangeSum"
elif config['KV']:
    config['NAME'] = "test_KV"
config['EPOCHS'] = 1000
config['train_size'] = 990000
config['test_size'] = 10000
config['depth'] = 0
config['no_comp'] = 10
config['learnable_depth'] = 0
config['no_filters'] = 1#config['n']//300
config['n'] = 500*(config['no_filters']**config['depth'])
config['k_th'] = 10
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
    config['in_dim'] = 10 
    config['lr']=0.001
    config['min_lr']=0.00005
    config['degree'] = 1
    config['batch_normalization'] = True
    config['filter_width1'] = 40
    config['filter_width2'] = 20 
    config['phi_no_layers'] = 8
    config['accuracy_threshold'] = 0.1

if config['NN']:
    config['out_dim'] = config['in_dim']
else:
    config['out_dim'] = 1
config['accuracy_mult_threshold'] = 0.1
config['batch_size'] = 1
config['on_hot_encode'] = False
config['no_processes'] = 2
config['entropy'] = True
config['leaf_no_samples'] = 0
config['query_type'] = 0
config['q_range'] = 1
config['agg_type'] = 2
config['MAX_VAL'] = 10
config['SAVE_PREDS'] = True
config['ONE_MODEL_PER_DIM'] = False
config['LOAD_MODEL'] = "-1"
config['data_loc'] = ""

py_loc = ''


settings=[]
settings.append({'filter_width1':60, 'filter_width2':30, 'phi_no_layers':3, 'depth':1, 'no_filters':2, 'n':1000, 'in_dim':25, 'batch_size':50, 'batch_normalization':True, 'k_th':100, 'EPOCHS':1000})


for setting in settings:
    name =config['NAME']
    for k, v in setting.items():
        config[k]=v
        name += '_'+str(v)
    
    os.system('mkdir tests')
    os.system('mkdir tests/'+ name)
    with open('tests/'+name+'/conf.json', 'w') as f:
        json.dump(config, f)

    command = 'cd tests/'+ name + ' && python '+py_loc+' conf.json 1 ' + name + ' > out.txt  & '
    os.system(command)
    print(name)

