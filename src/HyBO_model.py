from graph_model_BO import ChebConvmodel
from Runmodel import *
import numpy as np
import torch
from genericpath import exists
import site
from turtle import distance
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import os
import argparse
import sys
import numpy as np
import keras
import tensorflow as tf
# from me_data_BO import *
from Runmodel_BO_EI import embedding_data
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input,Flatten,concatenate
from tensorflow.keras import optimizers
from spektral.data import BatchLoader
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from spektral.transforms import GCNFilter
from spektral_me.loaders_me import BatchLoader_2feat
# from graph_model import GCNmodel,APPNPConvmodel,ARMAConvmodel,ChebConvmodel,GATConvmodel,GCSConvmodel
from spektral.layers import GCNConv, GlobalSumPool,GlobalMaxPool,APPNPConv,ARMAConv,ChebConv,GATConv,GCSConv
from tensorflow.keras.models import load_model
from tqdm import tqdm
from gensim.models import Word2Vec
from graph_model_BO import ChebConvmodel

# tf.config.experimental.set_virtual_device_configuration(
#     tf.config.experimental.list_physical_devices('GPU')[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
# )

custom_objects={'ChebConv': ChebConv,
                'GlobalSumPool':GlobalSumPool,
                'GATConv':GATConv,
                'APPNPConv':APPNPConv,
                'ARMAConv':ARMAConv,
                'GCNConv':GCNConv,
                'GlobalMaxPool':GlobalMaxPool,
                'GCSConv':GCSConv}

class Problem(object):
    def __init__(self, dimension, lower_bounds, upper_bounds):
        self.dimension = dimension
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        pass

class HyBO_model(object):
    def __init__(self,random_seed=None,n_points=10):
        self.num_continuous=2 #连续超参个数
        n_vertices=np.array([5]) #每个离散变量的节点数
        self.n_vertices=n_vertices
        self.num_discrete=self.n_vertices.shape[0] #离散超参个数
        self.random_seed = random_seed
        self.suggested_init = self.generate_random_points(n_points=n_points, random_seed=random_seed)#随机初始化
        # self.suggested_init=torch.tensor([[3,10]]) #随机初始化
        
        self.fourier_freq=[] #傅里叶频率,拉普拉斯矩阵的特征值
        self.fourier_basis=[] #傅里叶基础,拉普拉斯矩阵的特征向量
        self.adjacency_mat=[] #离散的图的邻接矩阵
        # self.problem=Problem(dimension=self.num_discrete+self.num_continuous,lower_bounds=[0,5],upper_bounds=[4,15])
        self.problem=Problem(dimension=self.num_discrete+self.num_continuous,lower_bounds=[0,0,0],upper_bounds=[4,1,18])
        for i in range(len(n_vertices)):
            n_v = n_vertices[i]
            adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(torch.ones(n_v - 1), 1)
            self.adjacency_mat.append(adjmat)
            laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
            # eigval, eigvec = torch.symeig(laplacian, eigenvectors=True)
            eigval, eigvec = torch.linalg.eigh(laplacian)
            self.fourier_freq.append(eigval)
            self.fourier_basis.append(eigvec)
    def fit_iter(self,iter,gpu_id,sitekind,hop,distance,data,label1,label0):
        # print("iter is:",iter)
        save_path_BO=f"../exp/BO/HyBO/{sitekind}"
        os.makedirs(f"{save_path_BO}",exist_ok=True)
        # print("args:",args)
        # print("gpu_id:",gpu_id)
        # print("sitekind:",sitekind)
        output=[]
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        My_model=ChebConvmodel(data=data,hop=hop,distance=distance,sitetype=sitekind,feature_chose=3)
        # model=self.construction_model(gpu_id=gpu_id,sitekind=sitekind,hop=hop,distance=distance,data=data)
        model=My_model.model()

        train_id,val_id,test_id=data_split(iter,label1,label0)
        # print("Run_Cheb里的data:",data)
        loader_train = BatchLoader_2feat(data[train_id], batch_size=64)
        loader_val = BatchLoader_2feat(data[val_id], batch_size=64)
        loader_test = BatchLoader_2feat(data[test_id], batch_size=64)
        callbacks_list=[
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(save_path_BO,'test_'+str(iter)+'.h5'),
                # filepath=f"../Result/{My_model.__class__.__name__}/model/{feature_name}_test_{i}.h5",
                monitor='val_accuracy',
                save_best_only=True,
            )
        ]
        model_fit_out=model.fit(loader_train.load(), steps_per_epoch=loader_train.steps_per_epoch, epochs=200,validation_data=loader_val.load(),validation_steps=loader_val.steps_per_epoch,callbacks=callbacks_list)
        print("------------------Testing------------")
        model = load_model(f"{save_path_BO}/test_{iter}.h5",custom_objects=custom_objects)
        output.append(My_model.evaluate(loader_test,iter,model))
        # print(f"threshold:{threshold} sp:{test_output[-1]} sn:{test_output[2]} AUC:{test_output[-2]} AP:{test_output[-3]}")
        return output

    def evaluate_hybo(self, x_unorder,sitetype,cuda,iter_num,ac_kind):
        hop=int(x_unorder[0])+1 #优化的超参数 跳数
        distance=x_unorder[-1]
        print(f"hop:{hop}")
        save_path_BO=f"../exp/BO/HyBO/{sitetype}/{ac_kind}"
        os.makedirs(f"{save_path_BO}",exist_ok=True)
        print(f"distance:{distance}")
        temp=[]
        print(f"sitetype:{sitetype}")
        data,hop_nouse,distance_nouse,label1,label0,x1=embedding_data(hop=hop,distance=distance,sitetype=sitetype,cuda=cuda)
        for i in range(iter_num):
            print(f"-------------------{i}---fit-----------------b----")
            temp=self.fit_iter(iter=i,gpu_id=cuda,sitekind=sitetype,hop=hop,distance=distance,data=data,label1=label1,label0=label0)
            print("temp:",temp)
        result=np.mean(temp,0)
        # print("")
        with open(f"{save_path_BO}/bo_out_AUC.txt",'a') as f:
            f.write(str(distance)+' '+str(hop)+' '+str(result))
            f.write("\n")
        print(f"test result:{result}")
        return -torch.tensor(result[4]).float(),hop,distance

    def sample_points(self, n_points, random_seed=None):
        if random_seed is not None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(random_seed)
        init_points = []
        for _ in range(n_points):
            random_point = []
            random_point.append(torch.randint(0, 5, (1,)))
            # random_point.append(torch.randint(0, 2, (1,)))
            random_point.append(torch.FloatTensor(1).uniform_(0, 1))
            for i in range(1):
                random_point.append(torch.FloatTensor(1).uniform_(5, 15))

            init_points.append(random_point)
        return torch.tensor(init_points).float()

    def generate_random_points(self, n_points, random_seed=None):
        return self.sample_points(n_points, random_seed=self.random_seed if random_seed is None else random_seed).float()

