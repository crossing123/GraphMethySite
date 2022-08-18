# GraphMethySite

## 以第一个赖氨酸数据集举例
## 数据预处理
proprocess_data.ipynb
##     生成蛋白质距离矩阵

##    根据蛋白质距离矩阵生成蛋白质邻接矩阵
    calculate_Adj.ipynb

##     将np转换为fasta数据格式
    np_to_fasta.ipynb
##     训练Word2Vec模型
    produce_word2vec.py
## 生成每一个样本的邻接矩阵、特征矩阵（Onehot编码、Word2Vec编码、Onehot+Word2Vec编码）
## 1/ Onehot编码
    One_hot.ipynb
## 2/ Word2Vec编码
    
## 3/ Onehot+Word2Vec编码

## 将数据处理成子图格式，embedding到模型中
    model_true.ipynb
