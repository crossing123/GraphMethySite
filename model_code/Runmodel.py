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
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
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
    if i ==4:
        test_id=label1[i*x1:(i+1)*x1]+label0[i*x1:(i+1)*x1]
        val_id=label1[(i+1)%5*x1:(i+2)%5*x1]+label0[(i+1)%5*x1:(i+2)%5*x1]
        train_id=label1[(i+2)%5*x1:i%5*x1]+label0[(i+2)%5*x1:i%5*x1]
    else:
        test_id=label1[i*x1:(i+1)*x1]+label0[i*x1:(i+1)*x1]
        val_id=label1[(i+1)*x1:(i+2)*x1]+label0[(i+1)*x1:(i+2)*x1]
        train_id=label1[0:i*x1]+label1[(i+2)*x1:]+label0[0:i*x1]+label0[(i+2)*x1:]
    return train_id,val_id,test_id

def Run_GCNmodel(data,hop,distance,sitetype,label1,label0,x1):
    for i in range(5):
        for feature_chose in range(1,4):
            if feature_chose==1:
                feature_name="Onehot"
            elif feature_chose==2:
                feature_name="Word2Vec"
            elif feature_chose==3:
                feature_name="O+W"
            train_id,val_id,test_id=data_split(i,label1,label0)
            # test_id=label1[i*x1:(i+1)*x1]+label0[i*x1:(i+1)*x1]
            # val_id=label1[(i+1)*x1:(i+2)*x1]+label0[(i+1)*x1:(i+2)*x1]
            # train_id=label1[0:i*x1]+label1[(i+2)*x1:]+label0[0:i*x1]+label0[(i+2)*x1:]

            loader_train = BatchLoader_2feat(data[train_id], batch_size=64)
            loader_val = BatchLoader_2feat(data[val_id], batch_size=64)
            loader_test = BatchLoader_2feat(data[test_id], batch_size=64)
            My_model=GCNmodel(data,hop,distance,sitetype,feature_chose)
            model=My_model.model()
            os.makedirs(os.path.join('../Result/',My_model.__class__.__name__,'model'),exist_ok=True)
            save_path=f'../Result'
            try:
                os.makedirs(save_path)
            except FileExistsError:
                print('FileExistsError')
            callbacks_list=[
                keras.callbacks.EarlyStopping(
                    monitor='accuracy',
                    patience=10,
                ),
                keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join('../Result/',My_model.__class__.__name__,'model',feature_name+'_test_'+str(i)+'.h5'),
                    monitor='val_accuracy',
                    save_best_only=True,
                )
            ]
            model_fit_out=model.fit(loader_train.load(), steps_per_epoch=loader_train.steps_per_epoch, epochs=200,validation_data=loader_val.load(),validation_steps=loader_val.steps_per_epoch,callbacks=callbacks_list)
            model = load_model(f"../Result/{My_model.__class__.__name__}/model/{feature_name}_test_{i}.h5",custom_objects=custom_objects)
            print("GCN模型最终结果",My_model.evaluate(loader_test,i,model))

def Run_ChebConvmodel(data,hop,distance,sitetype,label1,label0,x1):
    for feature_chose in range(1,4):
        for i in range(5):
            if feature_chose==1:
                feature_name="Onehot"
            elif feature_chose==2:
                feature_name="Word2Vec"
            elif feature_chose==3:
                feature_name="O+W"
            train_id,val_id,test_id=data_split(i,label1,label0)
            print("Run_Cheb里的data:",data)
            loader_train = BatchLoader_2feat(data[train_id], batch_size=64)
            loader_val = BatchLoader_2feat(data[val_id], batch_size=64)
            loader_test = BatchLoader_2feat(data[test_id], batch_size=64)
            My_model=ChebConvmodel(data=data,hop=hop,distance=distance,sitetype=sitetype,feature_chose=feature_chose)
            model=My_model.model()
            os.makedirs(os.path.join('../Result/',My_model.__class__.__name__,'model'),exist_ok=True)
            callbacks_list=[
                # keras.callbacks.EarlyStopping(
                #     monitor='accuracy',
                #     patience=10,
                # ),
                keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join('../Result/',My_model.__class__.__name__,'model',feature_name+'_test_'+str(i)+'.h5'),
                    # filepath=f"../Result/{My_model.__class__.__name__}/model/{feature_name}_test_{i}.h5",
                    monitor='val_accuracy',
                    save_best_only=True,
                )
            ]
            model_fit_out=model.fit(loader_train.load(), steps_per_epoch=loader_train.steps_per_epoch, epochs=200,validation_data=loader_val.load(),validation_steps=loader_val.steps_per_epoch,callbacks=callbacks_list)
            model = load_model(f"../Result/{My_model.__class__.__name__}/model/{feature_name}_test_{i}.h5",custom_objects=custom_objects)
            print(My_model.evaluate(loader_test,i,model))

def Run_GATConvmodel(data,hop,distance,sitetype,label1,label0,x1):
    for feature_chose in range(1,4):
        for i in range(5):
            if feature_chose==1:
                feature_name="Onehot"
            elif feature_chose==2:
                feature_name="Word2Vec"
            elif feature_chose==3:
                feature_name="O+W"
            train_id,val_id,test_id=data_split(i,label1,label0)
            # test_id=label1[i*x1:(i+1)*x1]+label0[i*x1:(i+1)*x1]
            # val_id=label1[(i+1)*x1:(i+2)*x1]+label0[(i+1)*x1:(i+2)*x1]
            # train_id=label1[0:i*x1]+label1[(i+2)*x1:]+label0[0:i*x1]+label0[(i+2)*x1:]
            # test_id=label1[i*x1:(i+1)*x1]+label0[i*x1:(i+1)*x1]
            # val_id=label1[i*x1:(i+1)*x1]+label0[i*x1:(i+1)*x1]
            # train_id=label1[0:i*x1]+label1[(i+1)*x1:]+label0[0:i*x1]+label0[(i+1)*x1:]

            loader_train = BatchLoader_2feat(data[train_id], batch_size=64)
            loader_val = BatchLoader_2feat(data[val_id], batch_size=64)
            loader_test = BatchLoader_2feat(data[test_id], batch_size=64)
            My_model=GATConvmodel(data,hop,distance,sitetype,feature_chose)
            os.makedirs(os.path.join('../Result/',My_model.__class__.__name__,'model'),exist_ok=True)
            model=My_model.model()
            callbacks_list=[
                # keras.callbacks.EarlyStopping(
                #     monitor='accuracy',
                #     patience=10,
                # ),
                keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join('../Result/',My_model.__class__.__name__,'model',feature_name+'_test_'+str(i)+'.h5'),
                    monitor='val_accuracy',
                    save_best_only=True,
                )
            ]
            model_fit_out=model.fit(loader_train.load(), steps_per_epoch=loader_train.steps_per_epoch, epochs=200,validation_data=loader_val.load(),validation_steps=loader_val.steps_per_epoch,callbacks=callbacks_list)
            model = load_model(f"../Result/{My_model.__class__.__name__}/model/{feature_name}_test_{i}.h5",custom_objects=custom_objects)
            print(My_model.evaluate(loader_test,i,model))

def Run_GCSConvmodel(data,hop,distance,sitetype,label1,label0,x1):
    for feature_chose in range(1,4):
        for i in range(5):
            if feature_chose==1:
                feature_name="Onehot"
            elif feature_chose==2:
                feature_name="Word2Vec"
            elif feature_chose==3:
                feature_name="O+W"            # test_id=label1[i*x1:(i+1)*x1]+label0[i*x1:(i+1)*x1]
            # val_id=label1[(i+1)*x1:(i+2)*x1]+label0[(i+1)*x1:(i+2)*x1]
            # train_id=label1[0:i*x1]+label1[(i+2)*x1:]+label0[0:i*x1]+label0[(i+2)*x1:]
            # test_id=label1[i*x1:(i+1)*x1]+label0[i*x1:(i+1)*x1]
            # val_id=label1[i*x1:(i+1)*x1]+label0[i*x1:(i+1)*x1]
            # train_id=label1[0:i*x1]+label1[(i+1)*x1:]+label0[0:i*x1]+label0[(i+1)*x1:]
            train_id,val_id,test_id=data_split(i,label1,label0)

            loader_train = BatchLoader_2feat(data[train_id], batch_size=64)
            loader_val = BatchLoader_2feat(data[val_id], batch_size=64)
            loader_test = BatchLoader_2feat(data[test_id], batch_size=64)
            My_model=GCSConvmodel(data,hop,distance,sitetype,feature_chose)
            model=My_model.model()
            os.makedirs(os.path.join('../Result/',My_model.__class__.__name__,'model'),exist_ok=True)
            callbacks_list=[
                # keras.callbacks.EarlyStopping(
                #     monitor='accuracy',
                #     patience=10,
                # ),
                keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join('../Result/',My_model.__class__.__name__,'model',feature_name+'_test_'+str(i)+'.h5'),
                    monitor='val_accuracy',
                    save_best_only=True,
                )
            ]
            model_fit_out=model.fit(loader_train.load(), steps_per_epoch=loader_train.steps_per_epoch, epochs=200,validation_data=loader_val.load(),validation_steps=loader_val.steps_per_epoch,callbacks=callbacks_list)
            model = load_model(f"../Result/{My_model.__class__.__name__}/model/{feature_name}_test_{i}.h5",custom_objects=custom_objects)
            print(My_model.evaluate(loader_test,i,model))

def Run_APPNPConvmodel(data,hop,distance,sitetype,label1,label0,x1):
    for feature_chose in range(1,4):
        for i in range(5):
            if feature_chose==1:
                feature_name="Onehot"
            elif feature_chose==2:
                feature_name="Word2Vec"
            elif feature_chose==3:
                feature_name="O+W"
            # test_id=label1[i*x1:(i+1)*x1]+label0[i*x1:(i+1)*x1]
            # val_id=label1[(i+1)*x1:(i+2)*x1]+label0[(i+1)*x1:(i+2)*x1]
            # train_id=label1[0:i*x1]+label1[(i+2)*x1:]+label0[0:i*x1]+label0[(i+2)*x1:]
            # test_id=label1[i*x1:(i+1)*x1]+label0[i*x1:(i+1)*x1]
            # val_id=label1[i*x1:(i+1)*x1]+label0[i*x1:(i+1)*x1]
            # train_id=label1[0:i*x1]+label1[(i+1)*x1:]+label0[0:i*x1]+label0[(i+1)*x1:]
            train_id,val_id,test_id=data_split(i,label1,label0)

            loader_train = BatchLoader_2feat(data[train_id], batch_size=64)
            loader_val = BatchLoader_2feat(data[val_id], batch_size=64)
            loader_test = BatchLoader_2feat(data[test_id], batch_size=64)
            My_model=APPNPConvmodel(data,hop,distance,sitetype,feature_chose)
            # os.makedirs(os.path.join('../Result/',My_model.__class__.__name__,'model'),exist_ok=True)
            model=My_model.model()
            # print(model.summary())
            # print("路径为：",os.path.join('../Result/',My_model.__class__.__name__,'model',feature_name+'_test_'+str(i)+'.h5'))
            callbacks_list=[
                # keras.callbacks.EarlyStopping(
                #     monitor='accuracy',
                #     patience=10,
                # ),
                #  keras.callbacks.ModelCheckpoint(
                #      filepath=os.path.join('../Result/',My_model.__class__.__name__,'model',feature_name+'_test_'+str(i)+'.h5'),
                #     monitor='val_accuracy',
                #     save_best_only=True,
                #  )
     
            ]
            model_fit_out=model.fit(loader_train.load(), steps_per_epoch=loader_train.steps_per_epoch, epochs=200,validation_data=loader_val.load(),validation_steps=loader_val.steps_per_epoch,callbacks=callbacks_list)
            #model = load_model(f"../Result/{My_model.__class__.__name__}/model/{feature_name}_test_{i}.h5",custom_objects=custom_objects)
            print(My_model.evaluate(loader_test,i,model))

def Run_ARMAConvmodel(data,hop,distance,sitetype,label1,label0,x1):
    for feature_chose in range(1,4):
        for i in range(5):
            if feature_chose==1:
                feature_name="Onehot"
            elif feature_chose==2:
                feature_name="Word2Vec"
            elif feature_chose==3:
                feature_name="O+W"
            # test_id=label1[i*x1:(i+1)*x1]+label0[i*x1:(i+1)*x1]
            # val_id=label1[(i+1)*x1:(i+2)*x1]+label0[(i+1)*x1:(i+2)*x1]
            # train_id=label1[0:i*x1]+label1[(i+2)*x1:]+label0[0:i*x1]+label0[(i+2)*x1:]
            # test_id=label1[i*x1:(i+1)*x1]+label0[i*x1:(i+1)*x1]
            # val_id=label1[i*x1:(i+1)*x1]+label0[i*x1:(i+1)*x1]
            # train_id=label1[0:i*x1]+label1[(i+1)*x1:]+label0[0:i*x1]+label0[(i+1)*x1:]
            train_id,val_id,test_id=data_split(i,label1,label0)

            loader_train = BatchLoader_2feat(data[train_id], batch_size=64)
            loader_val = BatchLoader_2feat(data[val_id], batch_size=64)
            loader_test = BatchLoader_2feat(data[test_id], batch_size=64)
            My_model=ARMAConvmodel(data,hop,distance,sitetype,feature_chose)
            os.makedirs(os.path.join('../Result/',My_model.__class__.__name__,'model'),exist_ok=True)
            model=My_model.model()
            callbacks_list=[
                # keras.callbacks.EarlyStopping(
                #     monitor='accuracy',
                #     patience=10,
                # ),
                keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join('../Result/',My_model.__class__.__name__,'model',feature_name+'_test_'+str(i)+'.h5'),
                    monitor='val_accuracy',
                    save_best_only=True,
                )
            ]
            model_fit_out=model.fit(loader_train.load(), steps_per_epoch=loader_train.steps_per_epoch, epochs=200,validation_data=loader_val.load(),validation_steps=loader_val.steps_per_epoch,callbacks=callbacks_list)
            model = load_model(f"../Result/{My_model.__class__.__name__}/model/{feature_name}_test_{i}.h5",custom_objects=custom_objects)
            print(My_model.evaluate(loader_test,i,model))

# def Run_model(data,hop,distance,sitetype,label1,label0,x1,My_model):
#     os.makedirs(os.path.join('../Result/',My_model.__class__.__name__,'model'),exist_ok=True)
#     for feature_chose in range(1,4):
#         for i in range(5):
#             if feature_chose==1:
#                 feature_name="Onehot"
#             elif feature_chose==2:
#                 feature_name="Word2Vec"
#             elif feature_chose==3:
#                 feature_name="O+W"
#             # test_id=label1[i*x1:(i+1)*x1]+label0[i*x1:(i+1)*x1]
#             # val_id=label1[(i+1)*x1:(i+2)*x1]+label0[(i+1)*x1:(i+2)*x1]
#             # train_id=label1[0:i*x1]+label1[(i+2)*x1:]+label0[0:i*x1]+label0[(i+2)*x1:]
#             test_id=label1[i*x1:(i+1)*x1]+label0[i*x1:(i+1)*x1]
#             val_id=label1[i*x1:(i+1)*x1]+label0[i*x1:(i+1)*x1]
#             train_id=label1[0:i*x1]+label1[(i+1)*x1:]+label0[0:i*x1]+label0[(i+1)*x1:]

#             loader_train = BatchLoader_2feat(data[train_id], batch_size=64)
#             loader_val = BatchLoader_2feat(data[val_id], batch_size=64)
#             loader_test = BatchLoader_2feat(data[test_id], batch_size=64)
            
#             model=My_model.model()
#             callbacks_list=[
#                 # keras.callbacks.EarlyStopping(
#                 #     monitor='accuracy',
#                 #     patience=10,
#                 # ),
#                 keras.callbacks.ModelCheckpoint(
#                     filepath=os.path.join('../Result/',My_model.__class__.__name__,'model',feature_name+'_test_'+str(i)+'.h5'),
#                     monitor='val_accuracy',
#                     save_best_only=True,
#                 )
#             ]
#             model_fit_out=model.fit(loader_train.load(), steps_per_epoch=loader_train.steps_per_epoch, epochs=200,validation_data=loader_val.load(),validation_steps=loader_val.steps_per_epoch,callbacks=callbacks_list)
#             print(My_model.evaluate(loader_test,i,model))


def embedding_data(hop,distance,sitetype,cuda):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda 
    # config=tf.ConfigProto()
    # config.gpu
    # tf.config.set_logical_device_configuration([tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
    # import tensorflow as tf
    # gpus = tf.config.experimental.list_physical_devices('GPU')  
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #             # Memory growth must be set before GPUs have been initialized
    #         print(e)

    using_gpu_index = int(cuda) # 使用的 GPU 号码
    gpu_list = tf.config.experimental.list_physical_devices('GPU')
    if len(gpu_list) > 0:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpu_list[using_gpu_index], 
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12134)]
            )
        except RuntimeError as e:
            print(e)
    else:
        print("Got no GPUs")


    if sitetype=='K':
        posi_number=580
    elif sitetype=='R':
        posi_number=1816
    data=me_data_h_d(posi_number=posi_number,hop=hop,distance=distance,sitetype=sitetype)
    folder_address=f"../Result/"
    os.makedirs(folder_address,exist_ok=True)
    # try:
    #     os.mkdir(folder_address)
    # except FileExistsError:
    #     print('Folder_address is existing')
    # print('Data number: ',data.number)
    conut0=0
    count1=0
    for i in range(len(data)):
        if data[i]['y'][1]==0:
            conut0 +=1
        else:
            count1 +=1
    # print('0_label: ',conut0)
    # print('1_label: ',count1)
    x1=count1//5
    # print('x1:',x1)
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
    return data,hop,distance,sitetype,label1,label0,x1


def get_params():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--hops', type=int, default=3,help="The Number of Hop to extract subgraphs")
    parser.add_argument('-d','--distance', type=float, default=10.0,help="The Number of Distance to extract subgraphs")
    parser.add_argument('-t','--sitetype', type=str, default='K',help="The Number of Distance to extract subgraphs")
    parser.add_argument('-cuda','--gpu_id', type=str, default='2',help="id(s) for CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()
    return args
    # Run_APPNPConvmodel(data,hop,distance,sitetype,label1,label0,x1)


def Run(args):
    data,hop,distance,sitetype,label1,label0,x1=embedding_data(args['hops'],args['distance'],args['sitetype'],args['gpu_id'])
    # Run_GCNmodel(data,hop,distance,sitetype,label1,label0,x1)
    # # print("data.n_node_features1:",data.n_node_features1)
    Run_ChebConvmodel(data,hop,distance,sitetype,label1,label0,x1)
    # Run_GATConvmodel(data,hop,distance,sitetype,label1,label0,x1)
    # Run_GCSConvmodel(data,hop,distance,sitetype,label1,label0,x1)
    # Run_ARMAConvmodel(data,hop,distance,sitetype,label1,label0,x1)
    # Run_APPNPConvmodel(data,hop,distance,sitetype,label1,label0,x1)

if __name__ == '__main__':
    args=vars(get_params())
    Run(args)