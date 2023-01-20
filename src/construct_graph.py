import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.NeighborSearch import NeighborSearch
import Bio.PDB
import pandas as pd
import numpy as np
from tqdm import tqdm

def Dis_matrix(site_type='K'):
    count=0
    # save_path='/home/chentb/RAID_data/chentb_data/methylation_data/distance_matrix/'
    save_path=f'../dat/graph_information/PDB_Structure/distance_matrix/{site_type}_site/'
    os.makedirs(save_path,exist_ok=True)
    path=f"../dat/data_example/{site_type}_site"
    name_seq=np.load(f'{path}/me_name_seq_{site_type}_have_PDB_alphafold.npy',allow_pickle=True).item()
    for name in tqdm(name_seq):
        try :
            p = PDBParser()
            structure= p.get_structure(name,'/home/Users/gly/Alphafold_database/PDB/AF-'+name+'-F1-model_v1.pdb')
            # structure= p.get_structure(name,f'{save_path}AF-'+name+'-F1-model_v1.pdb')
        except FileNotFoundError: 
            continue
        else:
            for model in structure:
                for chain in model:
                    distance_matrix=np.zeros((len(chain),len(chain)))            
                    for i in range(1,len(chain)+1):
                        for j in range(i+1,len(chain)+1):
                            distance_matrix[i-1][j-1]=chain[i]["CA"]-chain[j]["CA"]
                            distance_matrix[j-1][i-1]=chain[i]["CA"]-chain[j]["CA"]
                        np.save(f'{save_path}{name}.npy',distance_matrix)
        # break 
    print(count)


def Adj_matrix(distance,site_kind):
    # site_kind="K"
    count=0
    path=f"../dat/data_example/{site_kind}_site/"
    print('Let\'s start working on the distance matrix...')
    name_seq = np.load(path+f'me_name_seq_{site_kind}_have_PDB_alphafold.npy',allow_pickle=True).item()
    # save_ori_path=f'../graph_information/PDB_Structure/distance_matrix/{site_kind}_site'
    save_path=f'../dat/graph_information/{site_kind}_site/Adj_matrix_{str(float(distance))}'
    os.makedirs(save_path,exist_ok=True)
    
    for name in tqdm(list(name_seq.keys())):
        if name+'.npy' not in os.listdir(save_path):
            try:
                posit_matrix=np.load(f"../dat/graph_information/PDB_Structure/distance_matrix/{site_kind}_site/{name}.npy") #your PDB save npy
            except FileNotFoundError: 
                print(name)
                count=count+1
                continue
            else:
                length=posit_matrix.shape[0]
                adj_matrix_bo=np.zeros([length,length])
                for i in range(length):
                    for j in range(i+1,length):
                        if posit_matrix[i][j]<=distance:
                            adj_matrix_bo[i][j]=1
                            adj_matrix_bo[j][i]=1
                np.save(save_path+'/'+name+'.npy',adj_matrix_bo)
    # print(count)


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

import os
from gensim.models import Word2Vec

def positive_data(distance,hop,site_kind): 
    # hop=float(hop)
    count=0 #calculate the number proteins that could not generate graphs
    model_path=f"../dat/data_example/{site_kind}_site/feature_model/"
    w2v_model = Word2Vec.load(f"{model_path}withX_word2vec_{site_kind}.model")
    embedding_matrix_word2vec = w2v_model.wv.vectors
    vocab_list = list(w2v_model.wv.vocab.keys())
    word_index = {word: index for index, word in enumerate(vocab_list)}
    path="../dat/data_example/"+site_kind+"_site"
    name_seq=np.load(f'{path}/me_name_seq_{site_kind}_have_PDB_alphafold.npy',allow_pickle=True).item()
    print('Start processing positive data')
    positive_no=np.load(path+'/me_positive_name_site_'+site_kind+'_have_PDB_alphafold.npy',allow_pickle=True).item()
    save_path='../dat/embedding_data/'+str(hop)+'_hop_'+str(float(distance))+'_'+site_kind+'/positive'
    try:
        os.makedirs(save_path+'/adj')
        os.makedirs(save_path+'/feat_onehot')
        os.makedirs(save_path+'/feat_word2vec')
        os.makedirs(save_path+'/seq')
    except FileExistsError:  
        print('FileExists')

    # positive_site=[]
    # temp=np.load(f'/home/Users/gly/gly_me/data_process/{site_kind}_site/me_positive_name_site_{site_kind}_have_PDB_alphafold.npy',allow_pickle=True).item()
    # for i,j in temp.items():
    #     for k in range(len(j)):
    #         positive_site.append(str(j[k]))
    # # print(positive_site)
    # print(name_seq)
    for name in tqdm(name_seq):
        try:
            adj_matrix=np.load(f'../dat/graph_information/PDB_Structure/distance_matrix/{site_kind}_site/{name}.npy')
        except FileNotFoundError: 
            count+=1
            print(name)
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


def negative_data(distance,hop,site_kind):
    # hop=float(hop)
    model_path=f"../dat/data_example/{site_kind}_site/feature_model/"
    w2v_model = Word2Vec.load(f"{model_path}withX_word2vec_{site_kind}.model")
    embedding_matrix_word2vec = w2v_model.wv.vectors
    vocab_list = list(w2v_model.wv.vocab.keys())
    word_index = {word: index for index, word in enumerate(vocab_list)}
    count=0
    path=f"../dat/data_example/{site_kind}_site/"
    name_seq=np.load(path+f'me_name_seq_{site_kind}_have_PDB_alphafold.npy',allow_pickle=True).item()
    print('Start processing negative data')
    negative_name_no=np.load(path+f'/me_negative_name_site_{site_kind}_have_PDB_alphafold.npy',allow_pickle=True).item()
    save_path='../dat/embedding_data/'+str(hop)+'_hop_'+str(float(distance))+'_'+site_kind+'/negative'
    try:
        os.makedirs(save_path+'/adj')
        os.makedirs(save_path+'/feat_onehot')
        os.makedirs(save_path+'/feat_word2vec')
        os.makedirs(save_path+'/seq')
    except FileExistsError:
        print('FileExistsError')
    for name in tqdm(name_seq):
        try:
            adj_matrix=np.load(f'../dat/graph_information/PDB_Structure/distance_matrix/{site_kind}_site/{name}.npy')
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


def get_params():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--hops', type=int, default=3,help="The Number of Hop to extract subgraphs")
    parser.add_argument('-d','--distance', type=float, default=10.0,help="The Number of Distance to extract subgraphs")
    parser.add_argument('-t','--sitetype', type=str, default='K',help="The Number of Distance to extract subgraphs")
    args = parser.parse_args()
    return args


def Run(args):
    Dis_matrix(site_type=args['sitetype'])
    Adj_matrix(args['distance'],site_kind=args['sitetype']) #defaul distance threshold is 10
    positive_data(args['distance'],args['hops'],args['sitetype'])
    negative_data(args['distance'],args['hops'],args['sitetype'])

if __name__ == '__main__':
    args=vars(get_params())
    Run(args)

