import subprocess
from sklearn.neighbors import NearestNeighbors
import os
import sys
from data_utils import get_nn_res
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

class Node:
    def __init__(self, dim, split_dim, fanout):
        self.dim = dim
        self.split_dim = split_dim
        self.fanout = fanout

        self.split_points = np.zeros((fanout-1,))
        self.children = None

    def set_split_point(self, x):
        max_size = 10000000
        if x.shape[0]>max_size:
            indx = np.argsort(x[:max_size, self.split_dim])
            size = max_size
        else:
            indx = np.argsort(x[:, self.split_dim])
            size = x.shape[0]

        if x[indx[0], self.split_dim] ==  x[indx[x.shape[0]-1], self.split_dim]:
            self.split_dim+=1
            self.set_split_point(x)
            return

        for i in range(self.fanout-1):
            self.split_points[i] = x[indx[(i+1)*size//self.fanout], self.split_dim]
            print(self.split_points[i])

    def get_params(self, file_name, node_id, get_base):
        if self.children == None:
            return ""

        if len(node_id) == 0:
            str_cnt = 'K\n'
        else:
            str_cnt = ''
        str_cnt += node_id+':'
        for i in range(self.fanout-1):
            str_cnt += str(self.split_points[i]) + ','
        str_cnt+='\n'

        for i in range(self.fanout):
            str_cnt+=self.children[i].get_params(file_name, node_id+str(i+1), get_base)
        return str_cnt
        
def train_model(DB, x, y, test_x, test_y, k_th, return_dist, dim, split_dim, fanout, min_dims, max_dims, leaf_size, processes, no_processes, no, base_name, path):
    if DB is not None:
        x = min_dims + (np.random.rand(leaf_size, dim)*(max_dims-min_dims))
        test_x = min_dims + np.random.rand(leaf_size, dim)*(max_dims-min_dims)
        y, test_y = get_nn_res(DB, x, test_x, k_th, return_dist)
        
    leaf = Node(dim, split_dim, fanout)

    if len(processes) == no_processes:
        node = os.environ['NODE_NAME']
        with open("../../running_pids"+node+".txt", 'a') as f:
            for p in processes:
                f.write(str(p.pid)+"\n")
        for p in processes:
            p.wait()
        processes=[]

    np.savetxt('queries'+str(no)+'.txt', x, delimiter=',', fmt='%.16f');
    np.savetxt('res'+str(no)+'.txt', y, delimiter=',', fmt='%.16f');
    np.savetxt('test'+str(no)+'_queries.txt', test_x, delimiter=',', fmt='%.16f');
    np.savetxt('test'+str(no)+'_res.txt', test_y, delimiter=',', fmt='%.16f');

    print(x.shape)
    print(test_x.shape)
    p = subprocess.Popen(["python", "fit_base.py", str(no), base_name, path])  
    processes.append(p)

    return leaf, processes

def get_child_res(i, min_dims, max_dims, x, root, fanout, DB, split_dim, test_x, y, test_y):
    min_dims_new = np.copy(min_dims)
    max_dims_new = np.copy(max_dims)
    if i == 0:
        less = x[:, split_dim] < root.split_points[i]
        indx =less 
        test_less = test_x[:, split_dim] < root.split_points[i]
        test_indx =test_less 
    elif i == fanout-1:
        more = x[:, split_dim] >= root.split_points[i-1]
        indx =more 
        test_more = test_x[:, split_dim] >= root.split_points[i-1]
        test_indx =test_more 
    else:
        less = x[:, split_dim] < root.split_points[i]
        more = x[:, split_dim] >= root.split_points[i-1]
        indx = less and more
        test_less = test_x[:, split_dim] < root.split_points[i]
        test_more = test_x[:, split_dim] >= root.split_points[i-1]
        test_indx = test_less and test_more
    x_i=x[indx, :]
    test_x_i=test_x[test_indx, :]
    y_i = None
    test_y_i = None
    if DB is None:
        print(y)
        print(y.shape)
        y_i=y[indx, :]
        test_y_i=test_y[test_indx, :]

    if i!= 0:
        min_dims_new[split_dim] = root.split_points[i-1]
    if i!= fanout-1:
        max_dims_new[split_dim] = root.split_points[i]
    return min_dims_new, max_dims_new, x_i, test_x_i, y_i, test_y_i

def build_tree(depth, fanout, dim, x, y, test_x, test_y, processes, base_name, path, no, DB, k_th, split_dim, leaf_size, min_dims, max_dims, return_dist, no_processes):
    if depth == 0:
        leaf, processes = train_model(DB, x, y, test_x, test_y, k_th, return_dist, dim, split_dim, fanout, min_dims, max_dims, leaf_size, processes, no_processes, no, base_name, path)
        no+=1
        return leaf, no, processes

    root = Node(dim, split_dim, fanout)
    root.set_split_point(x)
    #min_dims_new, max_dims_new, x_i, test_x_i, y_i, test_y_i = get_child_res(i, min_dims, max_dims, x, root, fanout, DB) 

    root.children = []
    for i in range(fanout):
        min_dims_new, max_dims_new, x_i, test_x_i, y_i, test_y_i = get_child_res(i, min_dims, max_dims, x, root, fanout, DB, split_dim, test_x, y, test_y) 
        child, no, processes = build_tree(depth-1, fanout, dim, x_i, y_i, test_x_i, test_y_i, processes, base_name, path+str(i+1), no, DB, k_th, (split_dim+1)%dim, leaf_size, min_dims_new, max_dims_new, return_dist, no_processes)
        root.children.append(child)

    return  root, no, processes


