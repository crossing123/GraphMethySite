from dis import dis
from spektral_me.data_me import Graph,Dataset,Graph_2feat,Dataset_2feat
from spektral.utils.convolution import gcn_filter
import os
import numpy as np
import random
from tqdm import tqdm

class me_data_h_d(Dataset_2feat):

    """
    A dataset of five random graphs.
    """
    def __init__(self, posi_number,hop,distance,sitetype, **kwargs):
        self.number=posi_number
        # self.number_0=posi_number
        self.class_name=self.__class__.__name__
        self.hop=hop
        self.distance=distance
        self.sitetype=sitetype
        self.save_path=f"../dat/embedding_model/me_{self.hop}_hop_{self.distance}_{sitetype}"
        self.output=[]
        super().__init__(**kwargs)
    def download(self):
        data = ...  # Download from somewhere
        # Create the directory
        if True:
            os.makedirs(self.save_path,exist_ok=True)
            # Write the data to file
            print('Loading the positive data...')
            positive_num=0
            data_ori_path='../dat/embedding_data/'
            data_path=f'{data_ori_path}{self.hop}_hop_{self.distance}_{self.sitetype}/positive/'
            matrixs_path=data_path+'adj/'
            feature_onehot_path=data_path+'feat_onehot/'
            feature_word2vec_path=data_path+'feat_word2vec/'
            seq_path=data_path+'seq/'
            os.makedirs(data_ori_path,exist_ok=True)
            for name in tqdm(os.listdir(matrixs_path)):
                a=np.load(matrixs_path+name)
                x1=np.load(feature_onehot_path+name)
                x2=np.load(feature_word2vec_path+name)
                seq=np.load(seq_path+name)
                y=[0,1]
                filename = os.path.join(self.save_path, f'graph_positive{positive_num}')
                np.savez(filename, x1=x1, x2=x2, a=a, y=y,seq=seq,name=name)
                positive_num+=1
            negative_num=0
            data_ori_path='../dat/embedding_data/'
            data_path=f'{data_ori_path}{self.hop}_hop_{self.distance}_{self.sitetype}/negative/'
            matrixs_path=data_path+'adj/'
            feature_onehot_path=data_path+'feat_onehot/'
            feature_word2vec_path=data_path+'feat_word2vec/'
            seq_path=data_path+'seq/'
            print('Loading the negative data...')
            for name in tqdm(os.listdir(matrixs_path)):
                a=np.load(matrixs_path+name)
                x1=np.load(feature_onehot_path+name)
                x2=np.load(feature_word2vec_path+name)
                seq=np.load(seq_path+name)
                y=[1,0]
                filename = os.path.join(self.save_path, f'graph_negative{negative_num}')
                np.savez(filename, x1=x1, x2=x2, a=a, y=y,seq=seq,name=name)
                negative_num+=1
            # print(a)
    def read(self):
        # We must return a list of Graph objects
        output = []
        print('Reading data...')
        postive_number_list=list(range(self.number))
        # negative_number_list=list(random.sample(range(0 , count), self.number_0))
        negative_number_list=postive_number_list

        for i in tqdm(postive_number_list):
            data = np.load(os.path.join(self.save_path, f'graph_positive{i}.npz'))
            output.append(
                Graph_2feat(x=[data['x1'],data['x2']], a=data['a'], y=data['y'],seq=data['seq'],name=data['name'])
            )

        for i in tqdm(negative_number_list):
            data = np.load(os.path.join(self.save_path, f'graph_negative{i}.npz'))
            output.append(
                Graph_2feat(x=[data['x1'],data['x2']], a=data['a'], y=data['y'],seq=data['seq'],name=data['name'])
            )
        print(type(output))
        return output