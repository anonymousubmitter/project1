import random
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import time
import subprocess
import json
import os
node = os.environ['NODE_NAME']

config={}
config['NN_dist'] = False
config['NN'] = False
config['RangeSum'] = True
config['KV'] = False
if config['NN_dist']:
    config['NAME'] = "test_NN_dist"
elif config['NN']:
    config['NAME'] = "test_NN"
elif config['RangeSum']:
    config['NAME'] = "test_RangeSum"
elif config['KV']:
    config['NAME'] = "test_KV"
config['EPOCHS'] = 50000
config['train_size'] = 5000000
config['test_size'] = 10000
config['depth'] = 0
config['no_comp'] = 100
config['learnable_depth'] = 0
config['no_filters'] = 1#config['n']//300
config['n'] = 500*(config['no_filters']**config['depth'])
config['k_th'] = 100
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
    config['in_dim'] = 25
    config['lr']=0.001
    config['min_lr']=0.0001
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
config['no_processes'] = 1
config['entropy'] = True
config['leaf_no_samples'] = 0
config['query_type'] = 0
config['q_range'] = 1
config['agg_type'] = 2
config['MAX_VAL'] = 10
config['SAVE_PREDS'] = True
config['ONE_MODEL_PER_DIM'] = False
config['LOAD_MODEL'] = "-1"
config['db_ag_col'] = 1
config['db_sel_col'] = 0
config['normalized_loss'] = False
config['NON_NULL'] = 0
config['pred_dim'] = 2
config['pred_dim_pairs'] = 10
config['with_angle'] = False
config['data_loc'] = "/tank/users/zeighami/project1/data/tpc1_all_norm_filtered.npy"

config['query_loc'] = ""

py_loc = '/tank/users/zeighami/project1/main.py'

settings=[]
settings.append({'filter_width1':60, 'filter_width2':30, 'phi_no_layers':5, 'depth':4, 'no_filters':2, 'n':2620573, 'in_dim':12*2, 'batch_size':50, 'batch_normalization':False, 'EPOCHS':5000, 'agg_type':3, 'q_range':1, 'db_sel_col':0, '1_act_f':'relu', '2_act_f':'relu',  'train_data_size':2620573, "only_begin":False, "res_val_partition":False, "no_leaves":0, 'train_size':30000, 'NON_NULL':0, "pred_dim":1, "pred_dim_pairs":10})

max_process = 1
procs = []
for setting in settings:
    name =config['NAME']
    for k, v in setting.items():
        config[k]=v
        name += '_'+str(v)
    
    os.system('mkdir tests')
    os.system('mkdir tests/'+ name)
    with open('tests/'+name+'/conf.json', 'w') as f:
        json.dump(config, f)

    command = 'cd tests/'+ name + ' && python -u '+py_loc+' conf.json 1 ' + name + ' > out.txt  '
    #os.system(command)
    print(name)
    p1 = subprocess.Popen(command, shell=True) 
    procs.append(p1)
    print(len(procs))
    while len(procs) == max_process:
        for i, p in enumerate(procs):
            poll = p.poll()
            if poll is not None:
                del procs[i]
        time.sleep(1)

while len(procs) > 0:
    for i, p in enumerate(procs):
        poll = p.poll()
        if poll is not None:
            del procs[i]
    time.sleep(1)

email=True
if email:
    s = smtplib.SMTP(host='smtp.gmail.com', port=587)
    s.starttls()
    with open("email_cred.txt", "r") as f:
        usr, password = f.readline().split(",")  
    #print(usr.strip())
    #print(password.strip())
    s.login(usr.strip(), password.strip())
    msg = MIMEMultipart()
    message = "Exp on "+str(node)+" finished"
    msg['From']='sepzeighami@gmail.com'
    msg['To']="zeighami@usc.edu"
    msg['Subject']="EXP FINISHED"
    msg.attach(MIMEText(message, 'plain'))
    s.send_message(msg,'sepzeighami@gmail.com','zeighami@usc.edu',)
