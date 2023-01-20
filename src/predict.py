#!/home/Users/gly/anaconda3/envs/GCN/bin/python
from genericpath import exists
import site
import matplotlib.pyplot as plt
import os
import os
import argparse
import sys
import numpy as np
import keras
import tensorflow as tf
from me_data import *
# from keras.backend.tensorflow_backend import set_session
from spektral.data import Graph,Dataset
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input,Flatten,concatenate
from tensorflow.keras import optimizers
from spektral.data import BatchLoader
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve, auc, classification_report
from spektral.transforms import GCNFilter
from spektral_me.loaders_me import BatchLoader_2feat
from graph_model import GCNmodel,APPNPConvmodel,ARMAConvmodel,ChebConvmodel,GATConvmodel,GCSConvmodel
from spektral.layers import GCNConv, GlobalSumPool,GlobalMaxPool,APPNPConv,ARMAConv,ChebConv,GATConv,GCSConv
from tensorflow.keras.models import load_model

custom_objects={'ChebConv': ChebConv,
                'GlobalSumPool':GlobalSumPool,
                'GATConv':GATConv,
                'APPNPConv':APPNPConv,
                'ARMAConv':ARMAConv,
                'GCNConv':GCNConv,
                'GlobalMaxPool':GlobalMaxPool,
                'GCSConv':GCSConv}

def data_split(i,label1,label0,fold=5):
    x1=len(label1)//fold
    if i==4:
        test_id=label1[i*x1:(i+1)*x1]+label0[i*x1:(i+1)*x1]
        val_id=label1[(i+1)%5*x1:(i+2)%5*x1]+label0[(i+1)%5*x1:(i+2)%5*x1]
        train_id=label1[(i+2)%5*x1:i%5*x1]+label0[(i+2)%5*x1:i%5*x1]
    else:
        test_id=label1[i*x1:(i+1)*x1]+label0[i*x1:(i+1)*x1]
        val_id=label1[(i+1)*x1:(i+2)*x1]+label0[(i+1)*x1:(i+2)*x1]
        train_id=label1[0:i*x1]+label1[(i+2)*x1:]+label0[0:i*x1]+label0[(i+2)*x1:]
    return train_id,val_id,test_id


def evaluate(loader,ite,model,feature_chose):
    step = 0
    true_label=[]
    pre_score=[]
    pre_label=[]
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)
        for i in target:
            if i[0]==1:
                true_label.append(0)
            else:
                true_label.append(1)
        for i in pred:
            pre_score.append(i[1])
        for i in pred:
            if i[0]>=0.5:
                pre_label.append(0)
            else:
                pre_label.append(1)
    # print("true_label",true_label)
    return true_label,pre_score,pre_label

def plot(true_label,pre_score,pre_label):
    # print("true_label",true_label)
    # print("pre_score",pre_score)
    # print("pre_label",pre_label)
    fpr, tpr, _ = roc_curve(true_label, pre_score)
    roc_auc = auc(fpr, tpr)
    print(classification_report(true_label, pre_label))
    print("The final evaluation indicators of AUC : ", roc_auc)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for ST')
    plt.legend(loc="lower right")
    plt.savefig('../vis/AUC.jpg')
    plt.show()

def embedding_data(data,hop,distance,sitetype,cuda):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda 
    label0=[]
    label1=[]
    positive_name=[]
    negative_name=[]
    true_list=[]
    pre_score_list=[]
    pre_list=[]
    for i in range(len(data)):
        if data[i]['y'][1]==1:
            label1.append(i)
            positive_name.append(str(data[i].name).split('.')[0])
        else:
            label0.append(i)
            negative_name.append(str(data[i].name).split('.')[0])
    #Start testing the model   
    for i in range(5):
        feature_chose=3
        feature_list=['Onehot','Word2Vec','O+W']
        feature_name=feature_list[feature_chose-1]
        train_id,val_id,test_id=data_split(i,label1,label0)
        loader_test = BatchLoader_2feat(data[test_id], batch_size=64)
        My_model=ChebConvmodel(data=data,hop=hop,distance=distance,sitetype=sitetype,feature_chose=feature_chose)
        model = load_model(f"../exp/Result/{My_model.__class__.__name__}/model/{feature_name}_test_{i}.h5",custom_objects=custom_objects)
        feature_name=feature_list[feature_chose-1]
        # print(My_model.evaluate(loader_test,i,model))
        true_label,pre_score,pre_label=evaluate(loader_test,i,model,feature_chose)
        true_list.append(true_label)
        pre_score_list.append(pre_score)
        pre_list.append(pre_label)
        # print(true_list) 2*232
        # print(np.array(true_list[0]).shape)
        new_true_list=(np.array(true_list)).flatten()
        new_pre_score_list=(np.array(pre_score_list)).flatten()
        new_pre_list=(np.array(pre_list)).flatten()
    # print(new_true_list)
    return new_true_list,new_pre_score_list,new_pre_list



def get_params():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--hops', type=int, default=3,help="The Number of Hop to extract subgraphs")
    parser.add_argument('-d','--distance', type=float, default=10.0,help="The Number of Distance to extract subgraphs")
    parser.add_argument('-t','--sitetype', type=str, default='K',help="The Number of Distance to extract subgraphs")
    parser.add_argument('-cuda','--gpu_id', type=str, default='0',help="id(s) for CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()
    return args
    # Run_APPNPConvmodel(data,hop,distance,sitetype,label1,label0,x1)

def Run(args):
    if args['sitetype']=='K':
        posi_number=580
    elif args['sitetype']=='R':
        posi_number=1816
    data=me_data_h_d(posi_number=posi_number,hop=args['hops'],distance=args['distance'],sitetype=args['sitetype'])
    new_true_list,new_pre_score_list,new_pre_list=embedding_data(data=data,hop=args['hops'],distance=args['distance'],sitetype=args['sitetype'],cuda=args['gpu_id'])  
    plot(new_true_list,new_pre_score_list,new_pre_list)

if __name__ == '__main__':
    args=vars(get_params())
    Run(args)