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

        print("split dim---------------", self.split_dim)
        print("indx", indx)
        num_in_lower = np.sum(x[:, self.split_dim] < x[indx[size//self.fanout], self.split_dim])
        num_in_upper = x.shape[0]-num_in_lower
        print("sel. points", x[:, self.split_dim] < x[indx[size//self.fanout], self.split_dim])
        print("count", num_in_lower)
        #num_in_lower = np.sum(x[:, self.split_dim] >= x[indx[size//self.fanout], self.split_dim])
        print("count", num_in_upper)
        print("x.shape", x.shape)
        print("min count", min(num_in_lower, num_in_upper))
        print("thresh",  0.2*x.shape[0])
        #if x[indx[0], self.split_dim] ==  x[indx[x.shape[0]-1], self.split_dim]:
        if min(num_in_lower, num_in_upper) < 0.2*x.shape[0]:
            self.split_dim=(self.split_dim+1)%x.shape[1]
            return self.set_split_point(x)

        for i in range(self.fanout-1):
            self.split_points[i] = x[indx[(i+1)*size//self.fanout], self.split_dim]
            #print("split point", i, self.split_points[i])
            #print("split dim",i , self.split_dim)
        return self.split_dim

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
        #x = min_dims + (np.random.rand(leaf_size, dim)*(max_dims-min_dims))
        #test_x = min_dims + np.random.rand(leaf_size, dim)*(max_dims-min_dims)
        print("Model No", no)
        print(max_dims)
        print(min_dims)
        min_dims = np.array([2.5, 0.5])
        max_dims = np.array([4.2, 2.7])
        q_range=0.004
        max_val=10
        x =  min_dims + (np.random.rand(leaf_size, dim)*(max_dims-min_dims))
        end_queries =  x+q_range*max_val
        #x = np.concatenate([begin_queries, end_queries], axis=2)
        y = np.array([np.sum([np.logical_and.reduce([np.logical_and(DB[:, d]>=x[i, d], DB[:, d]<end_queries[i, d]) for d in range(dim)])]) for i in range(x.shape[0])]).reshape((-1, 1))
        test_x =  min_dims + (np.random.rand(leaf_size, dim)*(max_dims-min_dims))
        end_queries =  test_x+q_range*max_val
        #test_x = np.concatenate([begin_queries, end_queries], axis=2)
        test_y = np.array([np.sum([np.logical_and.reduce([np.logical_and(DB[:, d]>=test_x[i, d], DB[:, d]<end_queries[i, d]) for d in range(dim)])]) for i in range(test_x.shape[0])]).reshape((-1, 1))
        #x = x.reshape(leaf_size, dim)
        #y = y.reshape(leaf_size, dim)
        #y, test_y = get_nn_res(DB, x, test_x, k_th, return_dist, "")
        
    leaf = Node(dim, split_dim, fanout)

    print("y.shape", y.shape)

    if len(processes) == no_processes:
        node = os.environ['NODE_NAME']
        with open("../../running_pids"+node+".txt", 'a') as f:
            for p in processes:
                f.write(str(p.pid)+"\n")
        for p in processes:
            p.wait()
        processes=[]

    np.save('queries'+str(no)+'.npy', x);
    np.save('res'+str(no)+'.npy', y);
    np.save('test'+str(no)+'_queries.npy', test_x);
    np.save('test'+str(no)+'_res.npy', test_y);

    #print(x.shape)
    #print(test_x.shape)
    p = subprocess.Popen(["python", "/tank/users/zeighami/project1/fit_base.py", str(no), base_name, path])  
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
    #print("indx", indx)
    print("x.shape split", x.shape)
    print("child", i, " ", np.sum(indx))
    print("split point", root.split_points[0])
    print("split dim" , root.split_dim)
    x_i=x[indx, :]
    test_x_i=test_x[test_indx, :]
    y_i = None
    test_y_i = None
    if DB is None:
        #print(y)
        #print(y.shape)
        y_i=y[indx, :]
        test_y_i=test_y[test_indx, :]

    if i!= 0:
        min_dims_new[split_dim] = root.split_points[i-1]
    if i!= fanout-1:
        max_dims_new[split_dim] = root.split_points[i]
    return min_dims_new, max_dims_new, x_i, test_x_i, y_i, test_y_i

def build_tree(depth, fanout, dim, x, y, test_x, test_y, processes, base_name, path, no, DB, k_th, split_dim, leaf_size, min_dims, max_dims, return_dist, no_processes):
    print("depth", depth, x.shape, y.shape)
    if depth == 0:
        #if no != 15:
        #    no+=1
        #    return 0, no, processes
        leaf, processes = train_model(DB, x, y, test_x, test_y, k_th, return_dist, dim, split_dim, fanout, min_dims, max_dims, leaf_size, processes, no_processes, no, base_name, path)
        no+=1
        return leaf, no, processes

    root = Node(dim, split_dim, fanout)
    #print("depth ", depth)
    split_dim = root.set_split_point(x)
    #min_dims_new, max_dims_new, x_i, test_x_i, y_i, test_y_i = get_child_res(i, min_dims, max_dims, x, root, fanout, DB) 

    root.children = []
    for i in range(fanout):
        min_dims_new, max_dims_new, x_i, test_x_i, y_i, test_y_i = get_child_res(i, min_dims, max_dims, x, root, fanout, DB, split_dim, test_x, y, test_y) 
        child, no, processes = build_tree(depth-1, fanout, dim, x_i, y_i, test_x_i, test_y_i, processes, base_name, path+str(i+1), no, DB, k_th, (split_dim+1)%dim, leaf_size, min_dims_new, max_dims_new, return_dist, no_processes)
        root.children.append(child)

    return  root, no, processes


