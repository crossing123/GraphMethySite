#!/home/Users/gly/anaconda3/envs/GCN/bin/python
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
from me_data_BO import *
from spektral.data import Graph,Dataset,BatchLoader
from bayes_opt import BayesianOptimization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input,Flatten,concatenate
from tensorflow.keras import optimizers
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from spektral.transforms import GCNFilter
# from data_process_backup.extract_graph_information.extract_graph_information import positive_data,negative_data,onehot_feature,get_index
from spektral_me.loaders_me import BatchLoader_2feat
# from graph_model import GCNmodel,APPNPConvmodel,ARMAConvmodel,ChebConvmodel,GATConvmodel,GCSConvmodel
from spektral.layers import GCNConv, GlobalSumPool,GlobalMaxPool,APPNPConv,ARMAConv,ChebConv,GATConv,GCSConv
from tensorflow.keras.models import load_model
from tqdm import tqdm
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt import UtilityFunction
from bayes_opt.util import load_logs
from gensim.models import Word2Vec
from graph_model_BO import ChebConvmodel

custom_objects={'ChebConv': ChebConv,
                'GlobalSumPool':GlobalSumPool,
                'GATConv':GATConv,
                'APPNPConv':APPNPConv,
                'ARMAConv':ARMAConv,
                'GCNConv':GCNConv,
                'GlobalMaxPool':GlobalMaxPool,
                'GCSConv':GCSConv}
save_path_BO=f'../exp/Result/BO'
cuda='1'


def get_index(sentence):
    word_index={'M': 0,
    'I': 1,
    'P': 2,
    'L': 3,
    'A': 4,
    'C': 5,
    'V': 6,
    'G': 7,
    'T': 8,
    'Y': 9,
    'D': 10,
    'S': 11,
    'Q': 12,
    'W': 13,
    'F': 14,
    'K': 15,
    'R': 16,
    'E': 17,
    'N': 18,
    'H': 19,
    'U': 20,
    'X': 21}
    sequence = []
    for word in sentence:
        try:
            sequence.append(word_index[word])
        except KeyError:
            pass
    return sequence

def onehot_feature(seq):
    X_data=np.array(list(map(get_index, [seq])))
    embedding_matrix_onehot=np.eye(22)
    return embedding_matrix_onehot[X_data[0],:]

def Adj_matrix(distance,site_kind='K'):
    path=f"../dat/data_example/{site_kind}_site/"
    print('Let\'s start working on the distance matrix...')
    name_seq = np.load(path+f'me_name_seq_{site_kind}_have_PDB_alphafold.npy',allow_pickle=True).item()
    save_ori_path=f'../dat/graph_information/BO'
    save_path=f"{save_ori_path}/{site_kind}_site/Adj_matrix"
    os.makedirs(save_ori_path,exist_ok=True)
    os.makedirs(save_path,exist_ok=True)
    for name in tqdm(list(name_seq.keys())):
        if name+'.npy' not in os.listdir(save_path):
            try:
                posit_matrix=np.load(f"/home/Users/gly/Alphafold_database/posit_matrix_new/{name}_F1.npy")
            except FileNotFoundError: 
                print(name)
                continue
            else:
                # print("distance:",type(distance))
                adj_matrix=(posit_matrix<distance.item()).astype(int)  
                np.save(save_path+'/'+name+'.npy',adj_matrix)

def bfs(adj, site, hop):
    output = []
    Q = []
    Q.append(site)
    for i in range(hop+1):
        temp=[]
        while Q != []:
            v = Q.pop(0)
            output.append(v)
            for n ,t in enumerate(adj[v]):
                if t==1 and n not in Q and n not in temp and n not in output:
                    temp.append(n)
        # print(temp)
        Q=Q+temp
        # print(Q)
    return output

def positive_data(hop,site_kind='K',npy_path="",name_seq="",model_path=""): 
    w2v_model = Word2Vec.load(f"{model_path}withX_word2vec_{site_kind}.model")
    embedding_matrix_word2vec = w2v_model.wv.vectors
    vocab_list = list(w2v_model.wv.vocab.keys())
    word_index = {word: index for index, word in enumerate(vocab_list)}
    
    count=0
    print('Start processing positive data')
    positive_no=np.load(npy_path+f'/me_positive_name_site_{site_kind}_have_PDB_alphafold.npy',allow_pickle=True).item()
    save_path=f'../dat/embedding_data/BO/{site_kind}/positive'

    os.makedirs(save_path+'/adj',exist_ok=True)
    os.makedirs(save_path+'/feat_onehot',exist_ok=True)
    os.makedirs(save_path+'/feat_word2vec',exist_ok=True)
    os.makedirs(save_path+'/seq',exist_ok=True)

    positive_site=[]
    temp=np.load(f'../dat/data_example/{site_kind}_site/me_positive_name_site_{site_kind}_have_PDB_alphafold.npy',allow_pickle=True).item()
    # print(temp)
    for i,j in temp.items():
        for k in range(len(j)):
            positive_site.append(str(j[k]))
    # print(positive_site)

    for name in tqdm(temp):
        try:
            adj_matrix=np.load(f'../dat/graph_information/BO/{site_kind}_site/Adj_matrix/{name}.npy',allow_pickle=True)
        except FileNotFoundError: 
            count+=1
            print("FileExistsError",name)
            continue
        else:
            # feature_matrix=np.load('/home/chentb/alphafold_database/feature_matrix_1/'+name+'.npy')
            sites=positive_no[name]
            for site in sites:
                try:
                    num=bfs(adj_matrix,site-1,hop)
                except IndexError:
                    print(name,site)
                seq=[]
                word2vec_seq='X'+name_seq[name]+'X'
                freture_ix_leaft=[]
                freture_ix_right=[]
                for i in num:
                    # print("seq:",seq)
                    # print("name:",name_seq[name][i])
                    try:
                        name_seq[name][i]
                    except IndexError:
                        print(name,i)
                    seq.append(name_seq[name][i])
                    ix_seq_leaft=word2vec_seq[i:i+2]
                    ix_seq_right=word2vec_seq[i+1:i+3]
                    freture_ix_leaft.append(word_index[ix_seq_leaft])
                    freture_ix_right.append(word_index[ix_seq_right])
                temp_feature_right=embedding_matrix_word2vec[freture_ix_right]
                temp_feature_leaft=embedding_matrix_word2vec[freture_ix_leaft]
                word2vec_feature=temp_feature_right+temp_feature_leaft

                temp_adj_matrix=adj_matrix[num][:,num]

                np.save(save_path+'/seq/'+name+'_'+str(site)+'.npy', num)
                np.save(save_path+'/adj/'+name+'_'+str(site)+'.npy',temp_adj_matrix)
                np.save(save_path+'/feat_onehot/'+name+'_'+str(site)+'.npy',onehot_feature(seq))
                np.save(save_path+'/feat_word2vec/'+name+'_'+str(site)+'.npy',word2vec_feature)
    print(count)

def negative_data(hop,site_kind='K',npy_path="",name_seq="",model_path=""):
    w2v_model = Word2Vec.load(f"{model_path}withX_word2vec_{site_kind}.model")
    embedding_matrix_word2vec = w2v_model.wv.vectors
    vocab_list = list(w2v_model.wv.vocab.keys())
    word_index = {word: index for index, word in enumerate(vocab_list)}
    count=0
    print('Start processing negative data')
    negative_name_no=np.load(npy_path+f'/me_negative_name_site_{site_kind}_have_PDB_alphafold.npy',allow_pickle=True).item()
    save_path=f'../dat/embedding_data/BO/{site_kind}/negative'
    os.makedirs(save_path+'/adj',exist_ok=True)
    os.makedirs(save_path+'/feat_onehot',exist_ok=True)
    os.makedirs(save_path+'/feat_word2vec',exist_ok=True)
    os.makedirs(save_path+'/seq',exist_ok=True)

    negative_site=[]
    temp=np.load(f'../dat/data_example/{site_kind}_site/me_negative_name_site_{site_kind}_have_PDB_alphafold.npy',allow_pickle=True).item()
    for i,j in temp.items():
        for k in range(len(j)):
            negative_site.append(str(j[k]))
    # print(negative_site)

    for name in tqdm(temp):
        try:
            adj_matrix=np.load(f'../dat/graph_information/BO/{site_kind}_site/Adj_matrix/{name}.npy')
        except FileNotFoundError: 
            count+=1
            continue
        else:
            sites=negative_name_no[name]
            for site in sites:
                try:
                    num=bfs(adj_matrix,site-1,hop)
                except IndexError:
                    print(name,site)
                seq=[]
                word2vec_seq='X'+name_seq[name]+'X'
                freture_ix_leaft=[]
                freture_ix_right=[]
                for i in num:
                    seq.append(name_seq[name][i])
                    ix_seq_leaft=word2vec_seq[i:i+2]
                    ix_seq_right=word2vec_seq[i+1:i+3]
                    freture_ix_leaft.append(word_index[ix_seq_leaft])
                    freture_ix_right.append(word_index[ix_seq_right])
                temp_feature_right=embedding_matrix_word2vec[freture_ix_right]
                temp_feature_leaft=embedding_matrix_word2vec[freture_ix_leaft]
                word2vec_feature=temp_feature_right+temp_feature_leaft
                temp_adj_matrix=adj_matrix[num][:,num]

                np.save(save_path+'/seq/'+name+'_'+str(site)+'.npy', num)
                np.save(save_path+'/adj/'+name+'_'+str(site)+'.npy',temp_adj_matrix)
                np.save(save_path+'/feat_onehot/'+name+'_'+str(site)+'.npy',onehot_feature(seq))
                np.save(save_path+'/feat_word2vec/'+name+'_'+str(site)+'.npy',word2vec_feature)
    print(count)



def data_split(i,label1,label0,fold=5):
    x1=len(label1)//fold
    if i ==4:
        test_id=label1[i*x1:(i+1)*x1]+label0[i*x1:(i+1)*x1]
        val_id=label1[(i+1)%5*x1:(i+2)%5*x1]+label0[(i+1)%5*x1:(i+2)%5*x1]
        train_id=label1[(i+2)%5*x1:i%5*x1]+label0[(i+2)%5*x1:i%5*x1]
    else:
        test_id=label1[i*x1:(i+1)*x1]+label0[i*x1:(i+1)*x1]
        val_id=label1[(i+1)*x1:(i+2)*x1]+label0[(i+1)*x1:(i+2)*x1]
        train_id=label1[0:i*x1]+label1[(i+2)*x1:]+label0[0:i*x1]+label0[(i+2)*x1:]
    return train_id,val_id,test_id


def Run_ChebConvmodel(data,hop,distance,site_kind,label1,label0):
    output=[]
    for i in range(5):
        feature_name="O+W"
        train_id,val_id,test_id=data_split(i,label1,label0)
        # print("Run_Cheb里的data:",data)
        loader_train = BatchLoader_2feat(data[train_id], batch_size=64)
        loader_val = BatchLoader_2feat(data[val_id], batch_size=64)
        loader_test = BatchLoader_2feat(data[test_id], batch_size=64)
        My_model=ChebConvmodel(data=data,hop=hop,distance=distance,sitetype=site_kind,feature_chose=3)
        model=My_model.model()
        os.makedirs(os.path.join(save_path_BO,'EI',My_model.__class__.__name__,'model'),exist_ok=True)
        callbacks_list=[
            # keras.callbacks.EarlyStopping(
            #     monitor='accuracy',
            #     patience=10,
            # ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(save_path_BO,'EI',My_model.__class__.__name__,'model','test_'+str(i)+'.h5'),
                # filepath=f"../Result/{My_model.__class__.__name__}/model/{feature_name}_test_{i}.h5",
                monitor='val_accuracy',
                save_best_only=True,
            )
        ]
        model_fit_out=model.fit(loader_train.load(), steps_per_epoch=loader_train.steps_per_epoch, epochs=200,validation_data=loader_val.load(),validation_steps=loader_val.steps_per_epoch,callbacks=callbacks_list)
        model = load_model(f"{save_path_BO}/EI/{My_model.__class__.__name__}/model/test_{i}.h5",custom_objects=custom_objects)
        output.append(My_model.evaluate(loader_test,i,model))
        print(My_model.evaluate(loader_test,i,model))
    out=np.mean(output,0)
    with open(f"{save_path_BO}/EI/bo_out_AUC_{site_kind}.txt",'a') as f:
        f.write(str(distance)+' '+str(hop)+' '+str(out))
        f.write("\n")
    # return 0.5*out[3]+0.5*out[4]
    return out[4]

def embedding_data(hop,distance,sitetype,cuda):   #ekl函数
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda 
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(gpus[0], True)
    if sitetype=='K':
        posi_number=580
    elif sitetype=='R':
        posi_number=1816
    model_path=f"../dat/data_example/{sitetype}_site/feature_model/"
    npy_path=f"../dat/data_example/{sitetype}_site"
    name_seq=np.load(npy_path+f'/me_name_seq_{sitetype}_have_PDB_alphafold.npy',allow_pickle=True).item()
    # data=me_data_h_d(posi_number=posi_number,hop=hop,distance=distance,sitetype=sitetype)
    # assert type(hop) == int
    Adj_matrix(distance=distance,site_kind=sitetype)
    positive_data(hop=hop,npy_path=npy_path,name_seq=name_seq,model_path=model_path,site_kind=sitetype)
    negative_data(hop=hop,npy_path=npy_path,name_seq=name_seq,model_path=model_path,site_kind=sitetype)
    data=me_data_h_d(posi_number=posi_number,hop=hop,distance=distance,sitetype=sitetype)
    folder_address=f"../exp/Result/"
    os.makedirs(folder_address,exist_ok=True)
    conut0=0
    count1=0
    for i in range(len(data)):
        if data[i]['y'][1]==0:
            conut0 +=1
        else:
            count1 +=1
    x1=count1//5
    label0=[]
    label1=[]
    positive_name=[]
    negative_name=[]
    num=0
    for i in range(len(data)):
        if data[i]['y'][1]==1:
            label1.append(i)
            positive_name.append(str(data[i].name).split('.')[0])
        else:
            label0.append(i)
            negative_name.append(str(data[i].name).split('.')[0])
            num+=1
    return data,hop,distance,label1,label0,x1

def save(res,sitetype='K'):
    number=len(res)
    out_filename=f"{save_path_BO}/EI/bo_out_AUC_{sitetype}.txt"
    with open(out_filename,'a') as f:
        for i in range(number):
            output=[]
            output.append(i)
            output.append(res[i]['target'])
            for j in res[i]['params'].keys():
                output.append(res[i]['params'][j])
            f.write(str(output))
            f.write("\n")

def function_to_be_optimized(distance,hop,sitekind='K',cuda=cuda):
    h = int(hop)
    # print(h,' ',distance)
    data,hop_nouse,distance_nouse,sitetype,label1,label0,x1=embedding_data(h,distance,sitekind,cuda)
    return Run_ChebConvmodel(data,hop,distance,sitetype,label1,label0)



def RunBO(sitetype='K',cuda=cuda):
    pbounds = {'distance': (0, 18),
                'hop':(1,5)
            }
    optimizer_by = BayesianOptimization(
        f=function_to_be_optimized,
        pbounds=pbounds,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    # os.makedirs(save_path+"/BO/bo_hop_dist_K.json",exist_ok=True)
    # load_logs(optimizer_by, logs=[f"{save_path_BO}/EI/bo_{sitetype}_out_AUC.json"])
    logger = JSONLogger(path=f'{save_path_BO}/EI/bo_{sitetype}_out_AUC_60.json')
    optimizer_by.subscribe(Events.OPTIMIZATION_STEP, logger)

# optimizer_by.probe(
#     params=[10.0],
#     lazy=True,
# )
    optimizer_by.maximize(
        init_points=5,
        n_iter=60,
        acq='ei',  #Acquisition functions
    )
    # optimizer_by.run_optimization(max_iter=30,init_points=5,acq='poi')
    # optimizer_by.plot_acquisition()
    # optimizer_by.plot_convergence()
    save(optimizer_by.res,sitetype=sitetype)

def get_params():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-t','--sitetype', type=str, default='K',help="The Number of Distance to extract subgraphs")
    parser.add_argument('-cuda','--gpu_id', type=str, default='0',help="id(s) for CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()
    return args


def Run(args):
    RunBO(args['sitetype'],args['gpu_id'])

if __name__ == '__main__':
    args=vars(get_params())
    Run(args)