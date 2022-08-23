from spektral.data import Graph,Dataset
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input,Flatten,concatenate,Add,BatchNormalization
from tensorflow.keras import optimizers
from spektral.layers import GCNConv, GlobalSumPool,GlobalMaxPool,APPNPConv,ARMAConv,ChebConv,GATConv,GCSConv
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
import keras
import numpy as np
import os

class Graphmodel():
    def __init__(self,data,hop,distance,sitetype,feature_chose) -> None:
        self.data=data
        # print("Graphmodel的data:",data)
        self.hop=hop
        self.distance=distance
        # print(self.distance)
        self.sitetype=sitetype
        self.feature_chose=feature_chose

    def metrics_out(self,true_label,pre_label,pre_score):
        ac=accuracy_score(true_label,pre_label)
        pr=precision_score(true_label,pre_label)
        re=recall_score(true_label,pre_label,pos_label=1)
        # sp=recall_score(true_label,pre_label,pos_label=0)
        f1=f1_score(true_label,pre_label)
        matrix=confusion_matrix(true_label,pre_label)
        TN=matrix[0][0]
        FP=matrix[0][1]
        FN=matrix[1][0]
        TP=matrix[1][1]
        # MCC=(TP*TN-TP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5)
        AUC=roc_auc_score(true_label,pre_score)
        return [ac,pr,re,f1,AUC]

    def evaluate(self,loader,ite,model):
        if self.feature_chose==1:
            feature_name="Onehot"
        elif self.feature_chose==2:
            feature_name="Word2Vec"
        elif self.feature_chose==3:
            feature_name="O+W"
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
            if step == loader.steps_per_epoch:
                os.makedirs(os.path.join('./Result',self.__class__.__name__,'middle_output'), exist_ok=True)
                os.makedirs(os.path.join('./Result',self.__class__.__name__,'middle_output',f'me_{self.hop}_hop_{self.distance}_{self.sitetype}'), exist_ok=True)
                
                if True:
                    # os.makedirs(os.path.join('../Result',self.__class__.__name__,'middle_output'), exist_ok=True)
                    # os.makedirs(os.path.join('../Result',self.__class__.__name__,'middle_output'), exist_ok=True)
                    with open (os.path.join('./Result',self.__class__.__name__,'middle_output',f'me_{self.hop}_hop_{self.distance}_{self.sitetype}',f'{feature_name}_evaluate_output.txt'),'a') as f:
                        f.write(str(self.metrics_out(true_label,pre_label,pre_score))+'\n')
                    np.save(os.path.join('./Result',self.__class__.__name__,'middle_output',f'me_{self.hop}_hop_{self.distance}_{self.sitetype}',f'{feature_name}_test_{ite}.npy'),pre_score)
                    np.save(os.path.join('./Result',self.__class__.__name__,'middle_output',f'me_{self.hop}_hop_{self.distance}_{self.sitetype}',f'{feature_name}_true_{ite}.npy'),true_label)
                return self.metrics_out(true_label,pre_label,pre_score)

class GCNmodel(Graphmodel):
    def __init__(self,data,hop,distance,sitetype,feature_chose) -> None:
        super().__init__(data,hop,distance,sitetype,feature_chose)
    def model(self):
        
        X_in1 = Input(shape=(None,self.data.n_node_features1))
        X_in2 = Input(shape=(None,self.data.n_node_features2))
        if self.feature_chose==1:
            X_in=Dense(128,use_bias=False)(X_in1)
        elif self.feature_chose==2:
            X_in=Dense(128,use_bias=False)(X_in2)
        elif self.feature_chose==3:
            X_in=Add()([Dense(128,use_bias=False)(X_in1),Dense(128,use_bias=False)(X_in2)])
        A_in = Input(shape=(None,None), sparse=True)
        X_1 = GCNConv(256, 'relu',use_bias=False,kernel_regularizer=keras.regularizers.l2(0.0002))([X_in, A_in])
        X_1=BatchNormalization()(X_1)
        X_2 = GCNConv(256, 'relu',use_bias=False,kernel_regularizer=keras.regularizers.l2(0.0002))([X_1, A_in])
        X_2=BatchNormalization()(X_2)
        X_3 = GCNConv(512, 'relu',use_bias=False,kernel_regularizer=keras.regularizers.l2(0.0002))([X_2, A_in])
        X_3=BatchNormalization()(X_3)

        X=concatenate([X_1,X_2,X_3])
        X=GlobalSumPool()(X)
        X=Dropout(0.3)(X)
        X=Dense(1024,'relu')(X)
        output=Dense(2,'softmax')(X)
        model = Model(inputs=[[X_in1,X_in2], A_in], outputs=output)
        model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.Adam(learning_rate=1e-4,beta_1=0.95,beta_2=0.95),
                metrics=['accuracy'])
        return model

class APPNPConvmodel(Graphmodel):
    def __init__(self,data,hop,distance,sitetype,feature_chose) -> None:
        super().__init__(data,hop,distance,sitetype,feature_chose)
    def model(self):
        X_in1 = Input(shape=(None,self.data.n_node_features1))
        X_in2 = Input(shape=(None,self.data.n_node_features2))
        if self.feature_chose==1:
            X_in=Dense(128,use_bias=False)(X_in1)
        elif self.feature_chose==2:
            X_in=Dense(128,use_bias=False)(X_in2)
        elif self.feature_chose==3:
            X_in=Add()([Dense(128,use_bias=False)(X_in1),Dense(128,use_bias=False)(X_in2)])
        A_in = Input(shape=(None,None), sparse=True)
        X_1 = APPNPConv(256,use_bias=False,kernel_regularizer=keras.regularizers.l2(0.0002))([X_in, A_in])
        X_1=BatchNormalization()(X_1)
        X_2 = APPNPConv(256,use_bias=False,kernel_regularizer=keras.regularizers.l2(0.0002))([X_1, A_in])
        X_2=BatchNormalization()(X_2)
        X_3 = APPNPConv(512,use_bias=False,kernel_regularizer=keras.regularizers.l2(0.0002))([X_2, A_in])
        X_3=BatchNormalization()(X_3)

        X=concatenate([X_1,X_2,X_3])
        X=GlobalSumPool()(X)
        X=Dropout(0.3)(X)
        X=Dense(1024,'relu')(X)
        output=Dense(2,'softmax')(X)
        model = Model(inputs=[[X_in1,X_in2], A_in], outputs=output)
        model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.Adam(learning_rate=1e-4,beta_1=0.95,beta_2=0.95),
                metrics=['accuracy'])
        return model

class ARMAConvmodel(Graphmodel):
    def __init__(self,data,hop,distance,sitetype,feature_chose) -> None:
        super().__init__(data,hop,distance,sitetype,feature_chose)
    def model(self):
        X_in1 = Input(shape=(None,self.data.n_node_features1))
        X_in2 = Input(shape=(None,self.data.n_node_features2))
        if self.feature_chose==1:
            X_in=Dense(128,use_bias=False)(X_in1)
        elif self.feature_chose==2:
            X_in=Dense(128,use_bias=False)(X_in2)
        elif self.feature_chose==3:
            X_in=Add()([Dense(128,use_bias=False)(X_in1),Dense(128,use_bias=False)(X_in2)])
        A_in = Input(shape=(None,None), sparse=True)
        X_1 = ARMAConv(256,iterations=2,use_bias=False,kernel_regularizer=keras.regularizers.l2(0.0002))([X_in, A_in])
        X_1=BatchNormalization()(X_1)
        X_2 = ARMAConv(256,iterations=2,use_bias=False,kernel_regularizer=keras.regularizers.l2(0.0002))([X_1, A_in])
        X_2=BatchNormalization()(X_2)
        X_3 = ARMAConv(512,iterations=2,use_bias=False,kernel_regularizer=keras.regularizers.l2(0.0002))([X_2, A_in])
        X_3=BatchNormalization()(X_3)

        X=concatenate([X_1,X_2,X_3])
        X=GlobalSumPool()(X)
        X=Dropout(0.3)(X)
        X=Dense(1024,'relu')(X)
        output=Dense(2,'softmax')(X)
        model = Model(inputs=[[X_in1,X_in2], A_in], outputs=output)
        model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.Adam(learning_rate=1e-4,beta_1=0.95,beta_2=0.95),
                metrics=['accuracy'])
        return model

class ChebConvmodel(Graphmodel):
    def __init__(self,data,hop,distance,sitetype,feature_chose) -> None:
        super().__init__(data=data,hop=hop,distance=distance,sitetype=sitetype,feature_chose=feature_chose)
    def model(self):
        # print("ChebConvmodel的data:",self.data)
        X_in1 = Input(shape=(None,self.data.n_node_features1))
        X_in2 = Input(shape=(None,self.data.n_node_features2))
        # print("跳数:",self.hop)Gr
        if self.feature_chose==1:
            X_in=Dense(128,use_bias=False)(X_in1)
        elif self.feature_chose==2:
            X_in=Dense(128,use_bias=False)(X_in2)
        elif self.feature_chose==3:
            X_in=Add()([Dense(128,use_bias=False)(X_in1),Dense(128,use_bias=False)(X_in2)])
        # X_in1 = Input(shape=(None,self.data.n_node_features1))
        # X_in2 = Input(shape=(None,self.data.n_node_features2))
        A_in = Input(shape=(None,None), sparse=True)
        X_1 = ChebConv(256,k=3,activation='relu',use_bias=False,kernel_regularizer=keras.regularizers.l2(0.0002))([X_in, A_in])
        X_1=BatchNormalization()(X_1)
        X_2 = ChebConv(256,k=3,activation='relu',use_bias=False,kernel_regularizer=keras.regularizers.l2(0.0002))([X_1, A_in])
        X_2=BatchNormalization()(X_2)
        X_3 = ChebConv(512,k=3,activation='relu',use_bias=False,kernel_regularizer=keras.regularizers.l2(0.0002))([X_2, A_in])
        X_3=BatchNormalization()(X_3)

        X=concatenate([X_1,X_2,X_3])
        X=GlobalSumPool()(X)
        X=Dropout(0.6)(X) #初始Dropout是0.3
        X=Dense(1024,'relu')(X)
        output=Dense(2,'softmax')(X) 
        model = Model(inputs=[[X_in1,X_in2], A_in], outputs=output)
        model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.Adam(learning_rate=1e-4,beta_1=0.95,beta_2=0.95),
                metrics=['accuracy'])
        return model

class GATConvmodel(Graphmodel):
    def __init__(self,data,hop,distance,sitetype,feature_chose) -> None:
        super().__init__(data,hop,distance,sitetype,feature_chose)
    def model(self):
        X_in1 = Input(shape=(None,self.data.n_node_features1))
        X_in2 = Input(shape=(None,self.data.n_node_features2))
        if self.feature_chose==1:
            X_in=Dense(128,use_bias=False)(X_in1)
        elif self.feature_chose==2:
            X_in=Dense(128,use_bias=False)(X_in2)
        elif self.feature_chose==3:
            X_in=Add()([Dense(128,use_bias=False)(X_in1),Dense(128,use_bias=False)(X_in2)])
        A_in = Input(shape=(None,None), sparse=True)
        X_1 = GATConv(256,use_bias=False,kernel_regularizer=keras.regularizers.l2(0.0002))([X_in, A_in])
        X_1=BatchNormalization()(X_1)
        X_2 = GATConv(256,use_bias=False,kernel_regularizer=keras.regularizers.l2(0.0002))([X_1, A_in])
        X_2=BatchNormalization()(X_2)
        X_3 = GATConv(256,use_bias=False,kernel_regularizer=keras.regularizers.l2(0.0002))([X_2, A_in])
        X_3=BatchNormalization()(X_3)

        X=concatenate([X_1,X_2,X_3])
        X=GlobalSumPool()(X)
        X=Dropout(0.3)(X)
        X=Dense(1024,'relu')(X)
        output=Dense(2,'softmax')(X)
        model = Model(inputs=[[X_in1,X_in2], A_in], outputs=output)
        model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.Adam(learning_rate=1e-4,beta_1=0.95,beta_2=0.95),
                metrics=['accuracy'])
        return model

class GCSConvmodel(Graphmodel):
    def __init__(self,data,hop,distance,sitetype,feature_chose) -> None:
        super().__init__(data,hop,distance,sitetype,feature_chose)
    def model(self):
        X_in1 = Input(shape=(None,self.data.n_node_features1))
        X_in2 = Input(shape=(None,self.data.n_node_features2))
        if self.feature_chose==1:
            X_in=Dense(128,use_bias=False)(X_in1)
        elif self.feature_chose==2:
            X_in=Dense(128,use_bias=False)(X_in2)
        elif self.feature_chose==3:
            X_in=Add()([Dense(128,use_bias=False)(X_in1),Dense(128,use_bias=False)(X_in2)])
        A_in = Input(shape=(None,None), sparse=True)
        X_1 = GCSConv(256,use_bias=False,kernel_regularizer=keras.regularizers.l2(0.0002))([X_in, A_in])
        X_1=BatchNormalization()(X_1)
        X_2 = GCSConv(256,use_bias=False,kernel_regularizer=keras.regularizers.l2(0.0002))([X_1, A_in])
        X_2=BatchNormalization()(X_2)
        X_3 = GCSConv(256,use_bias=False,kernel_regularizer=keras.regularizers.l2(0.0002))([X_2, A_in])
        X_3=BatchNormalization()(X_3)

        X=concatenate([X_1,X_2,X_3])
        X=GlobalSumPool()(X)
        X=Dropout(0.4)(X)
        X=Dense(1024,'relu')(X)
        output=Dense(2,'softmax')(X)
        model = Model(inputs=[[X_in1,X_in2], A_in], outputs=output)
        model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.Adam(learning_rate=1e-4,beta_1=0.95,beta_2=0.95),
                metrics=['accuracy'])
        return model