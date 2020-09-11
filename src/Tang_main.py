# !/usr/bin/python
# -*- coding: UTF-8 -*-
import argparse
import os
import numpy as np
import random,math,time
import tensorflow as tf

from sklearn.metrics import roc_auc_score

def parse_args():

    parser = argparse.ArgumentParser(description="Run heer.")
    parser.add_argument('--dimensions', type=int, default=32,
	                    help='Number of free vectors dimensions. Default is 32.')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
	                    help='Learning Rate. Default is 0.0005.')
    parser.add_argument('--Pdim', type=int, default=12,
                    	help='Number of free project dimensions. Default is 12.')
    parser.add_argument("--K", type=int, default=10, help='K-nearest-neighbors. Default is 10.')
    parser.add_argument('--total_iter', type=int, default=4000,
                    	help='total_iter')
    
    parser.add_argument('--num_Comp', type=int, default=1351,
                    	help='total_iter')
    parser.add_argument('--dim_Comp', type=int, default=167,
                    	help='total_iter')
    parser.add_argument('--num_Kin', type=int, default=188,
                    	help='total_iter')

    return parser.parse_args()

def KNN(args,Sim_matr):
    row_matr = np.eye(Sim_matr.shape[0])
    for i in range(Sim_matr.shape[0]):
        max_row_index = np.argsort(Sim_matr[i,])[(len(Sim_matr[i,])-args.K):len(Sim_matr[i,])]
        for j in range(len(max_row_index)):
            row_matr[i,max_row_index[j]] = Sim_matr[i,max_row_index[j]]   
    return row_matr

def atten(args,matr1,matr2,matr_Y,flag): 
    if flag == "L":
        aT = tf.Variable(tf.truncated_normal([args.num_Kin,2*args.dimensions],dtype=tf.float32))
    elif flag == "T":
        aT = tf.Variable(tf.truncated_normal([args.num_Comp,2*args.dimensions], dtype=tf.float32))
    arrTemp_all = []
    for i in range(matr_Y.shape[0]):
        aa = tf.tile(tf.transpose([matr_Y[i,]]), [1,aT.shape[1]])  
        bb = aa * aT
        cc = bb * tf.concat([tf.tile([matr1[i,]], [matr_Y.shape[1],1]), matr2], axis = 1)
        dd = tf.reduce_sum(cc, axis = 1)
        arrTemp_all.append(tf.nn.softmax(tf.nn.leaky_relu(dd, alpha=0.2, name=None)))
    return arrTemp_all
    
def model_train(args,data_cK,data_cS,data_cF,test0,test1,MD_matOrigin):
    tf_A = tf.placeholder(tf.float32, data_cF.shape)
    tf_cS = tf.placeholder(tf.float32, data_cS.shape)
    tf_Y = tf.Variable(tf.truncated_normal([data_cK.shape[1], args.dimensions],stddev = 0.1,dtype=tf.float32))
    Yture = tf.placeholder(tf.float32, data_cK.shape)    
    
    D1 = tf.placeholder(tf.float32, data_cS.shape)
    
    Ww1 = tf.Variable(tf.truncated_normal([args.dim_Comp,args.dimensions],dtype=tf.float32))
    W1 = tf.Variable(tf.truncated_normal([data_cF.shape[1],data_cF.shape[1]],stddev=0.1,dtype=tf.float32))
    W12 = tf.Variable(tf.truncated_normal([data_cF.shape[1], data_cF.shape[1]], stddev=0.1, dtype=tf.float32))
    Wx1 = tf.Variable(tf.truncated_normal([data_cF.shape[1]+args.dimensions,data_cF.shape[1]],stddev=0.1,dtype=tf.float32))
    Wy1 = tf.Variable(tf.truncated_normal([data_cF.shape[1]+args.dimensions,args.dimensions],stddev=0.1,dtype=tf.float32))   
    Px = tf.Variable(tf.truncated_normal([data_cF.shape[1],args.Pdim], stddev=0.1,dtype=tf.float32))
    Py = tf.Variable(tf.truncated_normal([args.dimensions,args.Pdim], stddev=0.1,dtype=tf.float32))
    tf_alpha1 = tf.Variable(tf.nn.softmax(tf.truncated_normal([data_cK.shape[0],data_cK.shape[1]],stddev=0.1,dtype=tf.float32)))

    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(tf_Y))
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(Ww1))
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W1))
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W12))
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(Wx1))
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(Wy1))
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(Px))
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(Py))
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(tf_alpha1))

    D = np.zeros((data_cS.shape))
    data_cS2 = np.identity(data_cS.shape[0]) + data_cS
    for i in range(data_cS2.shape[0]):
        D[i,i] = math.pow(np.sum(data_cS2[i,]),-0.5)
    aa = tf.matmul(tf.matmul(D1,tf_cS),D1)

    X1 = tf.nn.relu(tf.matmul(tf.matmul(aa,tf_A),W1))
    X1_2 = tf.nn.relu(tf.matmul(tf.matmul(aa, X1), W12))

    X_att1 = tf.matmul(X1_2,Ww1)
    Layer1_atten1 = atten(args,X_att1,tf_Y,Yture,flag = "L")
    Yhet1 = tf.nn.relu(tf.matmul(tf.transpose(Layer1_atten1), X1_2))  
    Yrep1 = tf.sigmoid(tf.matmul(tf.concat([tf_Y, Yhet1], axis=1),Wy1))  
    Xhet1 = tf.nn.relu(tf.matmul(tf_alpha1, Yrep1))   
    Xrep1 = tf.sigmoid(tf.matmul(tf.concat([X1_2, Xhet1], axis=1),Wx1))

    Ypre = tf.matmul(tf.matmul(Xrep1,Px),tf.matmul(tf.transpose(Py),tf.transpose(Yrep1)))
    loss = tf.reduce_sum(tf.square(tf.subtract(Yture, Ypre)))   
    optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)
    init = tf.global_variables_initializer()
    
    # Start training
    with tf.Session() as sess:
        sess.run(init)
        feed_dict={tf_A:data_cF,tf_cS:data_cS2,Yture:data_cK,D1:D}
        y_=MD_matOrigin[test0,test1]
        list_train_loss = []
        for num_iter in range(args.total_iter):
            tic = time.time()
            sess.run(optimizer,feed_dict=feed_dict)
            train_loss=sess.run(loss,feed_dict=feed_dict)

            training_time = time.time() - tic
            list_train_loss.append(train_loss)
            reconstruct_v=sess.run(Ypre,feed_dict=feed_dict)
            y_pred=reconstruct_v[test0,test1]
            auc_score = roc_auc_score(y_,y_pred)

            if(num_iter%20==0):
                print('[TRN] iter = %03i, cost = %3.5e, AUC=%.5g,  (%3.2es)'% \
                      (num_iter, list_train_loss[-1], auc_score,  training_time))
        print("iter finish!")

def rowNorm(mat):
    tmp_mat = np.zeros((mat.shape[0],mat.shape[1]))
    for i in range(mat.shape[0]):
        if np.sum(mat[i,]) != 0:
           tmp_mat[i,] = mat[i,]/np.sum(mat[i,])
    return tmp_mat

def read_data(path):
    data = []
    for line in open(path, 'r'):
        ele = line.strip().split(" ")
        tmp = []
        for e in ele:
            if e != '':
                tmp.append(float(e))
        data.append(tmp)
    return data

if __name__ == '__main__':
    args = parse_args()
    data_path = os.path.join(os.path.dirname(os.getcwd()),"data\Tang")

    data_cK = read_data(data_path+'\\Tang_Matrix_Compound_Kinase_2.txt')
    data_cS = read_data(data_path+'\\Tang_matrix_ComComSim3.txt')
    data_cF = read_data(data_path+'\\Tang_Compound_feature166_2.txt')

    data_cK = np.array(data_cK)
    data_cS = np.array(data_cS)
    data_cF = np.array(data_cF)
    
    data_cS = rowNorm(data_cS)
    data_cS = KNN(args, data_cS)

    pos_position = np.zeros(shape=(int(np.sum(data_cK)),2))
    tmp_pos = 0
    tmp_arr = []
    for a in range(data_cK.shape[0]):
        for b in range(data_cK.shape[1]):
            if data_cK[a,b] == 1:
                pos_position[tmp_pos,0] = a
                pos_position[tmp_pos,1] = b
                tmp_arr.append(tmp_pos)
                tmp_pos +=1
                

    random.shuffle(tmp_arr)
    tep_pos_set = tmp_arr
    num_tep = math.floor(len(tep_pos_set)*0.2)
    t = 5
    
    for x in range(t):
        data_cK_new = np.zeros(shape = data_cK.shape)
        for i in range(data_cK.shape[0]):
            for j in range(data_cK.shape[1]):
                data_cK_new[i,j] = data_cK[i,j]      
        
        for j in range((x*num_tep),((x+1)*num_tep)):
            data_cK_new[int(pos_position[tep_pos_set[j],0]),int(pos_position[tep_pos_set[j],1])] = 0 
        
        test0 = []
        test1 = []
        for a in range(data_cK_new.shape[0]):
            for b in range(data_cK_new.shape[1]):
                if data_cK_new[a,b] == 0:
                    test0.append(a)
                    test1.append(b)

        model_train(args,data_cK_new,data_cS,data_cF,test0,test1,data_cK)
       




























