import time
import bisect
import datetime
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn import datasets
import multiprocessing
import json
import time

def std_w_empty(a):
    if a.shape[0] == 0:
        return 0
    return np.std(a)

def sum_w_empty(a):
    if a.shape[0] == 0:
        return 0
    return np.sum(a)

def median_w_empty(a):
    if a.shape[0] == 0:
        return 0
    return np.median(a)

def mean_w_empty(a):
    if a.shape[0] == 0:
        return 0
    return np.mean(a)


def calc_with_angle(x):
    x,  keys, vals, dim, agg_type = x

    getmask = lambda keys, x, d: np.logical_and(np.tan(x[-1])*keys[:, 0]+x[d+2] - np.tan(x[-1])*x[d]<keys[:, 1], (1/np.tan(x[-1]))*keys[:, 0]+x[d+2] - (1/np.tan(x[-1]))*x[d]<keys[:, 1])

    #slope1 = x[i, -1]
    #slope2 = 1/x[i, -1]
    #l1_mask = slope1*keys[:, 0]+x[i, 2] - slope1*x[i, 0]<keys[:, 1]
    #l2_mask = slope2*keys[:, 0]+x[i, 2] - slope2*x[i, 0]<keys[:, 1]

    #l3_mask = slope1*keys[:, 0]+x[i, 3] - slope1*x[i, 1]>keys[:, 1]
    #l4_mask = slope2*keys[:, 0]+x[i, 3] - slope2*x[i, 1]>keys[:, 1]

    return np.array([median_w_empty(vals[np.logical_and(getmask(keys, x[i], 0), getmask(keys, x[i], 1))]) for i in range(x.shape[0])]).reshape((-1, 1))

def calc(x):
    x,  keys, vals, dim, agg_type = x

    if agg_type == 2:
        return np.array([np.sum([np.logical_and.reduce([np.logical_and(keys[:, d]>=x[i, d, 0], keys[:, d]<x[i, d, 1]) for d in range(dim)])]) for i in range(x.shape[0])]).reshape((-1, 1))
        #res = np.array([np.sum([np.logical_and.reduce([np.logical_and(keys[:, d]>=queries[i, d, 0], keys[:, d]<queries[i, d, 1]) for d in range(dim)])]) for i in range(queries.shape[0])]).reshape((-1, 1))
    elif agg_type == 0:
        return np.array([std_w_empty(vals[np.logical_and.reduce([np.logical_and(keys[:, d]>=x[i, d, 0], keys[:, d]<x[i, d, 1]) for d in range(dim)])]) for i in range(x.shape[0])]).reshape((-1, 1))
        #res = np.array([std_w_empty(vals[np.logical_and.reduce([np.logical_and(keys[:, d]>=queries[i, d, 0], keys[:, d]<queries[i, d, 1]) for d in range(dim)])]) for i in range(queries.shape[0])]).reshape((-1, 1))
    elif agg_type == 1:
        return np.array([mean_w_empty(vals[np.logical_and.reduce([np.logical_and(keys[:, d]>=x[i, d, 0], keys[:, d]<x[i, d, 1]) for d in range(dim)])]) for i in range(x.shape[0])]).reshape((-1, 1))
        #res = np.array([mean_w_empty(vals[np.logical_and.reduce([np.logical_and(keys[:, d]>=queries[i, d, 0], keys[:, d]<queries[i, d, 1]) for d in range(dim)])]) for i in range(queries.shape[0])]).reshape((-1, 1))
    elif agg_type == 3:
        return np.array([sum_w_empty(vals[np.logical_and.reduce([np.logical_and(keys[:, d]>=x[i, d, 0], keys[:, d]<x[i, d, 1]) for d in range(dim)])]) for i in range(x.shape[0])]).reshape((-1, 1))
        #res = np.array([sum_w_empty(vals[np.logical_and.reduce([np.logical_and(keys[:, d]>=queries[i, d, 0], keys[:, d]<queries[i, d, 1]) for d in range(dim)])]) for i in range(queries.shape[0])]).reshape((-1, 1))
    #return np.array([mean_w_empty(vals[np.logical_and.reduce([np.logical_and(keys[:, d]>=x[i, d, 0], keys[:, d]<x[i, d, 1]) for d in range(dim)])]) for i in range(x.shape[0])]).reshape((-1, 1))

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

def get_nn_res(DB, queries, test_queries, k_th, return_dist, query_loc, out_dim):

    if query_loc != "":
        all_res = np.load(query_loc+"_res.npy")
        res = all_res[:queries.shape[0]]
        test_res =all_res[-test_queries.shape[0]:]
        #res = np.genfromtxt(query_loc+"_res.txt", delimiter=",")[:queries.shape[0]]
        #test_res = np.genfromtxt(query_loc+"_res.txt", delimiter=",")[-test_queries.shape[0]:]
    else:
        #print(queries)
        nbrs = NearestNeighbors(n_neighbors=k_th, algorithm='ball_tree').fit(DB)

        pool = multiprocessing.Pool()

        # pool object with number of element
        startTime = time.time()

        no_threads = 30
        pool = multiprocessing.Pool(processes=no_threads)

        # input list

        # map the function to the list and pass
        # function and input list as arguments
        outputs = pool.map(nbrs.kneighbors, [queries[i*(queries.shape[0]//no_threads):(i+1)*(queries.shape[0]//no_threads)] for i in range(no_threads)])
        distances = np.concatenate([outputs[i][0] for i in range(no_threads)], axis=0)
        indices = np.concatenate([outputs[i][1] for i in range(no_threads)], axis=0)
        print(indices.shape)

        executionTime = (time.time() - startTime)
        print('Execution time in seconds: ' + str(executionTime))


        #distances, indices = nbrs.kneighbors(queries)
        if return_dist: 
            res = distances[:, k_th-1]
        else:
            if out_dim == 1:
                res = indices[:, k_th-1].reshape(-1)
            else:
                res = DB[indices[:, k_th-1].reshape(-1), :out_dim]
        print("res.shape", res.shape)

        distances, indices = nbrs.kneighbors(test_queries)
        if return_dist:
            test_res = distances[:, k_th-1]
        else:
            if out_dim == 1:
                test_res = indices[:, k_th-1].reshape(-1)
            else:
                test_res = DB[indices[:, k_th-1].reshape(-1), :out_dim]

    return res, test_res

def get_nn_data(n, dim, return_dist, train_size, test_size, k_th, no_comp, data_loc, with_res, max_val, query_loc, out_dim):
    if data_loc != "":
        specific_datasets = ["cifar100", "mnist", "cifar", "gowalla", "synthetic", "gist"]
        if data_loc in specific_datasets:
            DB, queries, test_queries, query_res, test_res = load_pretrained_data(test_size, dim, k_th, max_val, data_loc, train_size, n)
            if query_res is not None:
                return DB, queries, test_queries, query_res.reshape(train_size, -1), test_res.reshape(test_size, -1)
        else:
            #DB_all = np.genfromtxt(data_loc+"/DB_original2.txt", delimiter='\t')
            if data_loc[-3:] == "npy":
                DB_all = np.load(data_loc)
            else:
                DB_all = np.genfromtxt(data_loc, delimiter='\t')
            #DB_all = np.genfromtxt(data_loc, delimiter=',')
            np.random.shuffle(DB_all)
            DB_all = ((DB_all-np.min(DB_all, axis=0))/(np.max(DB_all, axis=0)-np.min(DB_all, axis=0))-0.5)*max_val
            DB = DB_all.reshape(-1, dim)[:n, :]
            np.random.shuffle(DB_all)
            queries = DB_all[:train_size, :]
            test_queries = DB_all[-test_size:, :]
    else:
        DB, queries, test_queries = gen_rand_data(max_val, no_comp, n, train_size, test_size, dim)

    if query_loc != "":
        all_qs = np.load(query_loc+"_queries.npy")
        #np.random.shuffle(all_qs)
        queries = all_qs[:train_size]
        test_queries = all_qs[-test_size:]

    test_res = None
    res = None
    if with_res:
        res, test_res = get_nn_res(DB, queries, test_queries, k_th, return_dist, query_loc, out_dim)
        mask = res > 1e-3
        queries = queries[mask]
        res = res[mask]
        mask = test_res > 1e-3
        test_queries = test_queries[mask]
        test_res = test_res[mask]
        print("TEST RES MEAN:"+str(np.mean(test_res)))
        print("RES MEAN:"+str(np.mean(res)))
        res = res.reshape(res.shape[0], -1)
        test_res = test_res.reshape(test_res.shape[0], -1)
        print("res.shape get res", res.shape)
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

def get_range_agg_queries_and_res(dim, train_size, max_val, q_range, keys, vals, agg_type, query_loc, non_null, data_size, only_begin, dim_pairs, with_angle):
    if query_loc != "":
        queries = np.genfromtxt(query_loc+"_queries.txt", delimiter=",")[:train_size]
        res = np.genfromtxt(query_loc+"_res.txt", delimiter=",")[:train_size]
        #res = np.clip(res, a_min=None, a_max=24*3600)/60
        return queries, res

    keys = keys[:data_size]
    vals = vals[:data_size]

    print(keys.shape)
    print(vals.shape)
    print(np.min(keys))
    print(np.min(vals))

    if only_begin:
        dim = dim*2

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
        all_dims = False
        if all_dims:
            queries = np.sort((np.random.rand(train_size, dim, 2)-0.5)*max_val, axis=-1)
        else:
            queries = np.zeros((train_size, dim, 2))
            queries[:, :, 0] = -max_val/2
            queries[:, :, 0] = queries[:, :, 0]+np.random.rand(train_size, dim)*0.0001
            queries[:, :, 1] = max_val/2
            queries[:, :, 1] = queries[:, :, 1]+np.random.rand(train_size, dim)*0.0001
            queries = queries.reshape((train_size, dim*2))
            size_per_dim = train_size//dim
            np.savetxt("qspre.txt", queries)
            #dim_all_range = np.concatenate([np.zeros((size_per_dim, 1))-max_val/2, np.zeros((size_per_dim, 1))+max_val/2], axis=1)
            for i, x in enumerate(dim_pairs):
                for curr_dim in x:
                    if q_range == 1:
                        curr_dim_queries = np.sort((np.random.rand(size_per_dim, 2)-0.5)*max_val, axis=-1)
                    else:
                        begin_queries =  (np.random.rand(size_per_dim, 1))*(max_val*(1-q_range))+(-1*max_val*0.5)
                        end_queries =  begin_queries+q_range*max_val
                        curr_dim_queries = np.concatenate([begin_queries, end_queries], axis=-1)
                    queries[size_per_dim*i:size_per_dim*(i+1), 2*curr_dim] = curr_dim_queries[:, 0]
                    queries[size_per_dim*i:size_per_dim*(i+1), 2*curr_dim+1] = curr_dim_queries[:, 1]
            np.savetxt("qspost.txt", queries)
            queries = queries.reshape((train_size, dim, 2))

        if with_angle:
            queries = queries.reshape((train_size, 2*dim))
            angles = np.random.rand(train_size, 1)*np.pi/2
            queries = np.concatenate([queries, angles], axis=1)


    pool = multiprocessing.Pool()
    no_threads = 20
    pool = multiprocessing.Pool(processes=no_threads)


    if with_angle:
        func = calc_with_angle
    else:
        func = calc

    
    if keys.shape[0] > 20*(10**6):
        res = func((queries, keys, vals, dim, agg_type))
    else:
        outputs = pool.map(func, [(queries[i*(queries.shape[0]//no_threads):(i+1)*(queries.shape[0]//no_threads)], keys, vals, dim, agg_type) for i in range(no_threads)])
        res = np.concatenate([outputs[i] for i in range(no_threads)], axis=0)

    if agg_type != 2:
        if non_null >= 0:
            if non_null == 0:
                mask = (res > 0).reshape((-1))
            else:
                counts = np.array([np.sum([np.logical_and.reduce([np.logical_and(keys[:, d]>=queries[i, d, 0], keys[:, d]<queries[i, d, 1]) for d in range(dim)])]) for i in range(queries.shape[0])]).reshape((-1, 1))
                mask = counts[:, 0]>non_null
            queries = queries[mask] 
            res = res[mask] 
    #res = np.clip(res, a_min=None, a_max=24*3600)/60

    if only_begin:
        queries =  queries[:, :, 0]
        queries = queries.reshape(queries.shape[0], dim)
    else:
        if not with_angle:
            queries = queries.reshape(queries.shape[0], dim*2)


    print("MEAN:"+str(np.mean(res)))

    return queries, res


def get_range_agg_data(n, dim, train_size, test_size, no_comp, q_range, agg_type, max_val, db_path, db_sel_col, db_ag_col, query_loc, non_null, train_data_size, only_begin, pred_dim, pred_dim_pairs, with_angle):
    dim = dim//2

    if db_path=="gmm":
        samples = []
        if no_comp == n:
            DB = (np.random.rand(n, dim+1)-0.5)*max_val
        else:
            for i in range(no_comp):
                mean = (np.random.rand(dim+1)-0.5)*max_val
                #sigma = np.diag(((np.random.rand(dim))*max_val)/10)
                sigma = datasets.make_spd_matrix(dim+1)*max_val/10
                samples.append(np.random.multivariate_normal(mean, sigma, n//no_comp))
            DB = np.concatenate(samples)
        vals = DB[:, -1]
        if np.min(vals) < 0:
            vals = vals - np.min(vals)
        DB = DB[:, :-1]
    else:
        if db_path[-3:] == "npy":
            db_all = np.load(db_path)
        else:
            db_all = np.genfromtxt(db_path, delimiter=",", max_rows=n)
        np.random.shuffle(db_all)
        #db = db_all[:, db_sel_col:db_sel_col+1]
        db = db_all[:, :-1]
        db = db[:, :dim]
        min_vals = np.min(db, axis=0)
        max_vals = np.max(db, axis=0)
        DB = ((db-min_vals)/(max_vals-min_vals)-0.5)*max_val
        vals = db_all[:, -1]

    keys = DB
    DB = np.append(keys, np.reshape(vals, (-1, 1)), 1)

    print(DB)


    if pred_dim == 1:
        dim_pairs = np.array(range(dim)).reshape((-1, 1))
    else:
        dim_pairs = np.zeros((pred_dim_pairs, pred_dim), dtype=int)
        for row in range(pred_dim_pairs):
            dim_pairs[row, :] = np.random.choice(dim, size=pred_dim, replace=False)
    print(keys.shape)
    queries, res = get_range_agg_queries_and_res(dim, train_size, max_val, q_range, keys, vals, agg_type, query_loc, non_null, train_data_size, only_begin, dim_pairs, with_angle)
    print(keys.shape)
    #test_queries, test_res = get_range_agg_queries_and_res(dim, test_size, max_val, q_range, keys, vals, agg_type, "", 0, n, only_begin)
    test_queries, test_res = get_range_agg_queries_and_res(dim, test_size, max_val, q_range, keys, vals, agg_type, query_loc, non_null, n, only_begin, dim_pairs, with_angle)

    return DB, queries, test_queries, res, test_res


def get_agg_data(n, dim, train_size, test_size, no_comp, data_loc, with_res, q_range, agg_type, max_val, db_path, db_sel_col, db_ag_col, query_loc, non_null, train_data_size, only_begin, pred_dim, pred_dim_pairs, with_angle):
    time_series = False
    if time_series:
        return  get_timeseries_data()

    return get_range_agg_data(n, dim, train_size, test_size, no_comp, q_range, agg_type, max_val, db_path, db_sel_col, db_ag_col, query_loc, non_null, train_data_size, only_begin, pred_dim, pred_dim_pairs, with_angle)


def get_data(n, dim, out_dim, NN, NN_dist, range_sum, train_size, test_size, one_hot_encode, k_th, no_comp, data_loc, with_res, q_range, agg_type, max_val, db_sel_col, db_ag_col, query_loc, non_null, train_data_size, only_begin, pred_dim, pred_dim_pairs, with_angle):
    if NN or NN_dist:
        return get_nn_data(n, dim, NN_dist, train_size, test_size, k_th, no_comp, data_loc, with_res, max_val, query_loc, out_dim)
    elif range_sum:
        return get_agg_data(n, dim, train_size, test_size, no_comp, data_loc, with_res, q_range, agg_type, max_val, data_loc, db_sel_col, db_ag_col, query_loc, non_null, train_data_size, only_begin, pred_dim, pred_dim_pairs, with_angle)
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


