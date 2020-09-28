import time
import bisect
import datetime
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import json

def load_cifar(n, train_size, test_size):
    DB = np.load("../NNNNDB/cifar_DB_32.npy", allow_pickle=True)
    queries = np.load("../NNNNDB/cifar_queries_32.npy", allow_pickle=True)
    test_queries = np.load("../NNNNDB/cifar_test_queries_32.npy", allow_pickle=True)
    np.random.shuffle(DB)
    np.random.shuffle(queries)
    np.random.shuffle(test_queries)
    DB = DB[:n, :]
    queries = queries[:train_size, :]
    test_queries = test_queries[:test_size, :]
    return DB, queries, test_queries, None, None

def load_gist(n, train_size, test_size):
    DB_all = np.load("/home/users/zeighami/NNDB/Data/gist/DB.npy")
    DB = DB_all[:n, :]
    queries = DB_all[:train_size, :]
    test_queries = DB_all[train_size:train_size+test_size, :]
    return DB, queries, test_queries, None, None

def load_gowalla(n, train_size, test_size):
    checkins = np.load("../NNPDB/checkins.npy", allow_pickle=True)
    np.random.shuffle(checkins)
    checkins = checkins[:train_size+n+test_size, 1:3]
    max_loc =  checkins.max(axis=0)
    min_loc =  checkins.min(axis=0)
    checkins = ((checkins - min_loc)/(max_loc-min_loc)-0.5)*max_val
    DB = checkins[:n, :]
    queries = checkins[n:train_size+n, :]
    test_queries = checkins[train_size+n:train_size+n+test_size, :]
    return DB, queries, test_queries, None, None

def gen_rand_data(max_val, no_comp, n, train_size, test_size, dim):
    normal_query = False
    samples = []
    queries = []
    test_queries = []
    for i in range(no_comp):
        mean = (np.random.rand(dim)-0.5)*max_val
        sigma = np.diag(((np.random.rand(dim))*max_val)/10)
        samples.append(np.random.multivariate_normal(mean, sigma, n//no_comp))
        if normal_query:
            queries.append(np.random.multivariate_normal(mean, sigma, train_size//no_comp))
            test_queries.append(np.random.multivariate_normal(mean, sigma, test_size//no_comp))
    DB = np.concatenate(samples)
    if normal_query:
        queries = np.concatenate(queries)
        test_queries = np.concatenate(test_queries)
    else:
        queries = (np.random.rand(train_size, dim)-0.5)*max_val
        test_queries = (np.random.rand(test_size, dim)-0.5)*max_val

    return DB, queries, test_queries

def load_synth(test_size, dim, k_th, max_val):
    DB = np.genfromtxt("/home/users/zeighami/NNDB/Data/synthetic/DB.txt", delimiter='\t')
    queries = np.genfromtxt("/home/users/zeighami/NNDB/Data/synthetic/train.txt", delimiter='\t')
    queries_res = np.genfromtxt("/home/users/zeighami/NNDB/Data/synthetic/train_res.txt")

    test_queries = (np.random.rand(test_size, dim)-0.5)*max_val
    print(DB.shape)
    print(queries.shape)
    print(queries_res.shape)
    print(test_queries.shape)

    res, test_res = get_nn_res(DB, test_queries, test_queries, k_th, True)
    return DB, queries, test_queries, res, test_res


def load_pretrained_data(test_size, dim, k_th, max_val, data_loc, train_size, n):
    if data_loc == "gowalla":
        return load_gowalla(n, train_size, test_size)
    elif data_loc == "cifar":
        return load_cifar(n, train_size, test_size)
    elif data_loc == "gist":
        return load_gist(n, train_size, test_size)
    elif data_loc == "synthetic":
        return load_synth(test_size, dim, k_th, max_val)
    elif data_loc == "cifar100" :
        no_classes = 100
        dim = 32
        no_channel = 3
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    else:
        no_classes = 10
        dim = 28
        no_channel = 1
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    DB = [[] for i in range(no_classes)]
    class_count = 0
    for i in range(x_train.shape[0]):
        if len(DB[y_train[i]]) == 0:
            DB[y_train[i]].append(x_train[i])
            class_count+=1
            if class_count == no_classes:
                break
    DB = np.array(DB).reshape(-1, dim*dim*no_channel)/255
    no_nine = False
    if no_nine:
        x_train = x_train[y_train!=9]
        y_train = y_train[y_train!=9]
        x_test = x_test[y_test == 9]
        y_test = y_test[y_test == 9]

    queries = x_train.reshape(-1, dim*dim*no_channel)/255
    test_queries = x_test.reshape(-1, dim*dim*no_channel)/255
    res = DB[y_train[:]]
    print(res.shape)
    test_res = DB[y_test[:]]
    if no_nine:
        DB = DB[:9]
    return DB, queries, test_queries, res.reshape(-1, out_dim), test_res.reshape(-1, out_dim)

def manually_cal_nn(DB, queries, test_queries):
    test_res = []
    res = []
    for i in range(test_queries.shape[0]):
        min_dist = 100000
        res_q = []
        for j in range(DB.shape[0]):
            if np.sum(np.square(test_queries[i, :] - DB[j, :])) < min_dist:
                min_dist = np.sum(np.square(test_queries[i, :] - DB[j, :]))
                res_q = DB[j, :]
        test_res.append(res_q)
    for i in range(queries.shape[0]):
        min_dist = 1000000
        res_q = []
        for j in range(DB.shape[0]):
            if np.sum(np.square(queries[i, :] - DB[j, :])) < min_dist:
                min_dist = np.sum(np.square(queries[i, :] - DB[j, :]))
                res_q = DB[j, :]
        res.append(res_q)
    res = np.array(res)
    test_res = np.array(test_res)

def get_nn_res(DB, queries, test_queries, k_th, return_dist):
    nbrs = NearestNeighbors(n_neighbors=k_th, algorithm='ball_tree').fit(DB)

    distances, indices = nbrs.kneighbors(queries)
    if return_dist: 
        res = distances[:, k_th-1]
    else:
        res = DB[indices[:, k_th-1].reshape(-1), :]

    distances, indices = nbrs.kneighbors(test_queries)
    if return_dist:
        test_res = distances[:, k_th-1]
    else:
        test_res = DB[indices[:, k_th-1].reshape(-1), :]

    return res, test_res

def get_nn_data(n, dim, return_dist, train_size, test_size, k_th, no_comp, data_loc, with_res, max_val):
    if data_loc != "":
        specific_datasets = ["cifar100", "mnist", "cifar", "gowalla", "synthetic", "gist"]
        if data_loc in specific_datasets:
            DB, queries, test_queries, query_res, test_res = load_pretrained_data(test_size, dim, k_th, max_val, data_loc, train_size, n)
            if query_res is not None:
                return DB, queries, test_queries, query_res.reshape(train_size, -1), test_res.reshape(test_size, -1)
        else:
            DB_all = np.genfromtxt(data_loc+"/DB.txt", delimiter='\t')
            DB_all = (DB_all/np.max(DB_all))*max_val
            DB = DB_all.reshape(-1, dim)[:n, :]
            queries = DB_all[:train_size, :]
            test_queries = DB_all[-test_size:, :]
    else:
        DB, queries, test_queries = gen_rand_data(max_val, no_comp, n, train_size, test_size, dim)

    test_res = None
    res = None
    if with_res:
        res, test_res = get_nn_res(DB, queries, test_queries, k_th, return_dist)
        print("TEST RES MEAN:"+str(np.mean(test_res)))
        print("RES MEAN:"+str(np.mean(res)))
        res = res.reshape(train_size, -1)
        test_res = test_res.reshape(test_size, -1)
    return DB, queries, test_queries, res, test_res 

def get_timeseries_data():
    DB=[]
    with open('../checkins.txt', 'r') as f:
        for line in f.readlines():
            record = line.split()
            t = record[4]+' ' +record[5]
            import datetime
            date_time = datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.000")
            base = datetime.datetime(2000, 1, 1) 
            seconds_since =float((date_time-base).total_seconds())
            r = [float(record[0]), float(record[1]), float(record[2]), seconds_since]
            if r[0] != 1338:
                break
            DB.append(r)
    DB = np.array(DB)[:, 1:]
    queries = DB[:, 2].reshape(-1, 1)
    res = DB[:, 1:2].reshape(-1, 1)
    q_min = queries.min(); q_max = queries.max()
    queries = ((queries - q_min)/(q_max-q_min))*2*np.pi
    res_min = res.min(); res_max = res.max()
    res = ((res - res_min)/(res_max-res_min)-0.5)*10
    test_queries = None
    test_res = None
    return DB, queries, test_queries, res, test_res

def get_range_agg_queries_and_res(dim, train_size, max_val, q_range, keys, vals, agg_type):
    def mean_w_empty(a):
        if a.shape[0] == 0:
            return 0
        return np.mean(a)
    def std_w_empty(a):
        if a.shape[0] == 0:
            return 0
        return np.std(a)

    synthetic = True
    if not synthetic:
        sorts = [np.sort(keys[:, i]) for i in range(dim)]
        if q_range == 1:
            queries_indx = np.sort(np.random.randint(0, DB.shape[0], (train_size, dim, 2)), axis=-1)
            min_indx = queries_indx[:, :, 0]
            max_indx = queries_indx[:, :, 1]
            begin_queries = np.concatenate([sorts[i][min_indx[:, i]].reshape(-1, 1) for i in range(dim)], axis=1)
            end_queries = np.concatenate([sorts[i][max_indx[:, i]].reshape(-1, 1) for i in range(dim)], axis=1)
        else:
            queries_indx = np.random.randint(low=0, high=int((1-q_range)*DB.shape[0]), size=(train_size, dim))
            begin_queries = np.concatenate([sorts[i][queries_indx[:, i]].reshape(-1, 1) for i in range(dim)], axis=1)
            end_queries = np.concatenate([sorts[i][queries_indx[:, i]+int(q_range*DB.shape[0])].reshape(-1, 1) for i in range(dim)], axis=1)
        queries = np.concatenate([begin_queries.reshape(train_size, dim, 1), end_queries.reshape(train_size, dim, 1)], axis=2)
    else:
        if q_range == 1:
            queries = np.sort((np.random.rand(train_size, dim, 2)-0.5)*max_val, axis=-1)
        else:
            begin_queries =  (np.random.rand(train_size, dim, 1))*(max_val*(1-q_range)*0.5-max_val*0.5)+(-1*max_val*0.5)
            end_queries =  begin_queries+q_range*max_val
            queries = np.concatenate([begin_queries, end_queries], axis=2)

    if agg_type == 2:
        res = np.array([np.sum([np.logical_and.reduce([np.logical_and(keys[:, d]>=queries[i, d, 0], keys[:, d]<queries[i, d, 1]) for d in range(dim)])]) for i in range(queries.shape[0])]).reshape((-1, 1))
    elif agg_type == 0:
        res = np.array([std_w_empty(vals[np.logical_and.reduce([np.logical_and(keys[:, d]>=queries[i, d, 0], keys[:, d]<queries[i, d, 1]) for d in range(dim)])]) for i in range(queries.shape[0])]).reshape((-1, 1))
    elif agg_type == 1:
        res = np.array([mean_w_empty(vals[np.logical_and.reduce([np.logical_and(keys[:, d]>=queries[i, d, 0], keys[:, d]<queries[i, d, 1]) for d in range(dim)])]) for i in range(queries.shape[0])]).reshape((-1, 1))
    queries = queries.reshape(train_size, dim*2)
    print("MEAN:"+str(np.mean(res)))

    return queries, res


def get_range_agg_data(n, dim, train_size, test_size, no_comp, q_range, agg_type, max_val):
    dim = dim//2

    samples = []
    for i in range(no_comp):
        mean = (np.random.rand(dim)-0.5)*max_val
        sigma = np.diag(((np.random.rand(dim))*max_val)/10)
        samples.append(np.random.multivariate_normal(mean, sigma, n//no_comp))
    DB = np.concatenate(samples)

    keys = DB
    vals = np.random.rand(n)
    DB = np.append(keys, np.reshape(vals, (-1, 1)), 1)

    queries, res = get_range_agg_queries_and_res(dim, train_size, max_val, q_range, keys, vals, agg_type)
    test_queries, test_res = get_range_agg_queries_and_res(dim, train_size, max_val, q_range, keys, vals, agg_type)

    return DB, queries, test_queries, res, test_res


def get_agg_data(n, dim, train_size, test_size, no_comp, data_loc, with_res, q_range, agg_type, max_val):
    time_series = False
    if time_series:
        return  get_timeseries_data()

    return get_range_agg_data(n, dim, train_size, test_size, no_comp, q_range, agg_type, max_val)


def get_data(n, dim, out_dim, NN, NN_dist, range_sum, train_size, test_size, one_hot_encode, k_th, no_comp, data_loc, with_res, q_range, agg_type, max_val):
    if NN or NN_dist:
        return get_nn_data(n, dim, NN_dist, train_size, test_size, k_th, no_comp, data_loc, with_res, max_val)
    elif range_sum:
        return get_agg_data(n, dim, train_size, test_size, no_comp, data_loc, with_res, q_range, agg_type, max_val)
    else:
        queries = (np.random.rand(n, dim)-0.5)*2
        test_queries = None
        test_res = None
        if one_hot_encode:
            res = np.random.randint(0, 2, size=n)#*2-1
        else:
            #res = np.random.randint(0, 2, size=n)
            res = (np.random.rand(n, out_dim)-0.5)*max_val

        return queries, queries, test_queries, res, test_res


