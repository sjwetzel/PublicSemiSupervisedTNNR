# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 09:12:43 2019
@author: anonymous
"""
#%tensorflow_version 1.2
from __future__ import print_function
    
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
import tensorflow.keras
import itertools
from data.data import *
import sys
from progressbar import ProgressBar as PB
from SNN_helper import *
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,ReduceLROnPlateau,EarlyStopping
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Subtract,Input,Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,ReduceLROnPlateau,EarlyStopping
from sklearn.linear_model import LinearRegression
from tensorflow.keras import backend as K
import argparse
parser = argparse.ArgumentParser(description='TNN runner')
parser.add_argument('--dataset', help='Options are: bostonHousing, concreteData, energyEfficiency, proteinStructure, randomFunction', dest='dataset', default='bostonHousing')
parser.add_argument('--dataset_path', help='Relative path to datasets', dest='dataset_path', default='./data/')
parser.add_argument('--val_pct', help='Percentage of validation split.', dest='val_pct', default=0.05, type=float)
parser.add_argument('--test_pct', help='Percentage of test split.', dest='test_pct', default=0.05, type=float)
parser.add_argument('--l2', help='L2 Regularization weighting.', dest='l2', default=0.0, type=float)
parser.add_argument('--seed', help='Random seed', dest='seed', default=13, type=int)
parser.add_argument('--num', help='Number of datapoints for random function', default=1000, type=int)
parser = parser.parse_args()

Loop_weight=0.0
print(Loop_weight)
import time
print(parser.dataset)
start = time.time()

rmse_train_list=[]
rmse_val_list=[]
rmse_test_list=[]
rmse_test2_list=[]

### scaling factor 100,300,1000,3000,10000,30000,100000
n=1000
#(x_full, y_full) = getData(parser.dataset, parser.dataset_path)
#(x_full, y_full) = getData('randomFunction', './data/',n)
(x_full, y_full) = getData(parser.dataset, './data/',n)
#(x_full, y_full) = getData('testFunction', './data/',n)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
for iteration in range(25):
    parser.seed=iteration
    # set the seed
    print('Random seed:', parser.seed)
    
    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(parser.seed)
    
    # 3. Set the `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(parser.seed)
    
    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    #import tensorflow as tf
    tf.random.set_random_seed(parser.seed)
    
  

    
    #(x_train_single, y_train_single), (x_val_single, y_val_single), (x_test_single, y_test_single) = splitData(x_full, y_full, val_pct=parser.val_pct, test_pct=parser.test_pct)
    (x_train_single, y_train_single), (x_val_single, y_val_single), (x_test_single, y_test_single) = splitData(x_full, y_full, val_pct=0.1, test_pct=0.6,rand=True)
    print(x_train_single[0])
    
    ##### divide test set
    
    n_test_split=int(len(x_test_single)*5/6)
    x_test2_single=x_test_single[n_test_split:]
    y_test2_single=y_test_single[n_test_split:]    
    x_test_single=x_test_single[:n_test_split]
    y_test_single=y_test_single[:n_test_split]
    
    
    ##### center and normalize
    cn_transformer=CenterAndNorm()    
    
    
    x_train_single,y_train_single=cn_transformer.fittransform(x_train_single,y_train_single)
    x_val_single,y_val_single=cn_transformer.transform(x_val_single,y_val_single)
    x_test_single,y_test_single=cn_transformer.transform(x_test_single,y_test_single)
    x_test2_single,y_test2_single=cn_transformer.transform(x_test2_single,y_test2_single)

    ##########################################
    
    
    ############ NN parameters
    
    batch_size = 16
    epochs = 2000
    
    ############### create NN
    
    observer_a=Input(shape=(x_train_single.shape[-1],),name='observer_a')
    observer_b=Input(shape=(x_train_single.shape[-1],),name='observer_b')
    
    l2=parser.l2
    
    merged_layer = tensorflow.keras.layers.Concatenate()([observer_a, observer_b])
    
    merged_layer=Dense(128,activation='relu',kernel_regularizer=regularizers.l2(l2),activity_regularizer=regularizers.l2(l2))(merged_layer)
    merged_layer=Dense(128,activation='relu',name='iout',kernel_regularizer=regularizers.l2(l2),activity_regularizer=regularizers.l2(l2))(merged_layer)
    output=Dense(1,kernel_regularizer=regularizers.l2(l2),activity_regularizer=regularizers.l2(l2))(merged_layer)
    model = Model(inputs=[observer_a, observer_b], outputs=output)
    #model.summary()
    
    
    
    observer_c=Input(shape=(x_train_single.shape[-1],),name='observer_c')
    observer_d=Input(shape=(x_train_single.shape[-1],),name='observer_d')
    observer_e=Input(shape=(x_train_single.shape[-1],),name='observer_e')
    
    
    
    SNN1=model([observer_c,observer_d])
    SNN2=model([observer_d,observer_e])
    SNN3=model([observer_e,observer_c])
    
    tri_layer = tensorflow.keras.layers.Concatenate()([SNN1,SNN2,SNN3])
    
    
    TRImodel= Model(inputs=[observer_c, observer_d, observer_e], outputs=tri_layer)
    #TRImodel.summary()
    
    #### custom loss
    def custom_loss(y_true,y_pred):
        #eps=1.0e-6
        #y_pred=K.clip(y_pred,eps,1.0-eps)
        
        half=batch_size//2
        #y_true.set_shape([16,3])
        #y_pred.set_shape([16,3])
        y_case=tf.constant([1.,1.,1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.])
        y_case_odd=tf.constant([0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,1.,1.,1.,1.,1.,1.])
        #print(y_true.shape)
        #print((y_case*K.mean(y_pred,axis=-1)).shape)
        #return K.mean(K.square(tf.slice(y_pred-y_true,[0,0],[half,3]) ))+K.mean(K.square(K.mean(y_pred,axis=-1)))
        #return K.sqrt(K.mean(K.square(tf.slice(y_pred-y_true,[0,0],[half,3]) )))+100*K.abs(K.square(K.mean(y_pred,axis=-1)))
        #loss_A=K.mean(K.square(tf.slice(y_pred-y_true,[0,0],[half,3]) ),axis=-1)
        #loss_B=K.square(K.mean(tf.slice(y_pred,[half,0],[half,3]),axis=-1))
        
        loss_A=y_case*K.mean(K.square(y_pred - y_true), axis=-1)
        loss_B=Loop_weight*y_case_odd*K.square(K.mean(y_pred,axis=-1))      
        #return loss_A+loss_B
        #return y_case*K.mean(K.square(y_pred - y_true), axis=-1)+Loop_weight*y_case_odd*K.square(K.mean(y_pred,axis=-1))
        #return loss_A+Loop_weight*loss_B
        #return 0.1*K.sum(     (K.sum(y_pred,axis=0)-batch_size/2)**2             )      - K.sum(y_pred*K.log(y_pred))
        return K.in_train_phase(loss_A+loss_B, loss_A)
    
    # Let's train the model 
    TRImodel.compile(loss=custom_loss
                  ,optimizer=tensorflow.keras.optimizers.Adadelta(lr=1)
                  #,optimizer='rmsprop'
                 ,metrics=['mse']
                  )
    
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                  patience=int(1000/np.sqrt(n)+1), verbose=1,min_lr=0)
    
    
    early_stop= EarlyStopping(monitor='val_loss', patience=2*int(1000/np.sqrt(n)+1), verbose=0)
    mcp_save = ModelCheckpoint('mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min', verbose=0)
    history = TRImodel.fit_generator(triple_semi_supervised_generator_all_excluding_labelled_loops(x_train_single, y_train_single, np.concatenate((x_test_single,x_val_single),axis=0), batch_size),
                                     #triple_generator(x_train_single, y_train_single,batch_size),
                                    steps_per_epoch=len(x_train_single)*10/batch_size,
                                    epochs=epochs,
                                    validation_data=triple_semi_supervised_generator3(x_val_single, y_val_single, x_train_single, batch_size),
                                    #validation_data=triple_generator(x_val_single, y_val_single, batch_size),
                                    validation_steps=len(x_val_single)*100/batch_size,
                                    callbacks=[reduce_lr,early_stop, mcp_save],verbose=0)
    
    #TRImodel.load_weights('mdl_wts.hdf5')
    
    
    # Plot training & test loss values
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Val'], loc='upper left')
    # # plt.show()
    # plt.savefig('loss.pdf')
        

    
    #####################

    Y_pred_train=[]
    Y_pred_r_train=[]
    Y_pred_check_train=[]
    Y_median_train=[]
    Y_var_train=[]
    Y_mse_train=[]
    
    
    
    for i in range(len(x_train_single)):
        pair_B=np.array([x_train_single[i]]*len(x_train_single))
        diffA=model.predict([pair_B,x_train_single]).flatten()
        diffB=model.predict([x_train_single,pair_B]).flatten()
        Y_pred_train.append(np.average(0.5*diffA-0.5*diffB+y_train_single, weights=None))
        Y_pred_r_train.append(np.average(-diffB+y_train_single))
        Y_pred_check_train.append(np.var(0.5*diffA+0.5*diffB))
        #Y_pred_check_train.append(np.average(np.abs(model.predict([pair_B,x_train_single])+model.predict([x_train_single,pair_B]))))
        Y_median_train.append(np.median(diffA+y_train_single))
        Y_var_train.append(np.var(0.5*diffA-0.5*diffB+y_train_single))
        Y_mse_train.append((Y_pred_train[i]-y_train_single[i])**2)
    
    
    Y_pred_train=cn_transformer.inversetransformY(np.array(Y_pred_train))
    Y_pred_r_train=cn_transformer.inversetransformY(np.array(Y_pred_r_train))
    Y_pred_check_train=np.array(Y_pred_check_train)*(cn_transformer.Ymax)
    Y_median_train=cn_transformer.inversetransformY(np.array(Y_median_train))
    Y_var_train=np.array(Y_var_train)*(cn_transformer.Ymax)**2
    Y_mse_train=np.array(Y_mse_train)*(cn_transformer.Ymax)**2
    Y_self_check_train=np.abs(np.array(model.predict([x_train_single,x_train_single]))).flatten()*(cn_transformer.Ymax)
    
    #####################

    Y_pred_val=[]
    Y_pred_r_val=[]
    Y_pred_check_val=[]
    Y_median_val=[]
    Y_var_val=[]
    Y_mse_val=[]
    
    
    
    for i in range(len(x_val_single)):
        pair_B=np.array([x_val_single[i]]*len(x_train_single))
        diffA=model.predict([pair_B,x_train_single]).flatten()
        diffB=model.predict([x_train_single,pair_B]).flatten()
        Y_pred_val.append(np.average(0.5*diffA-0.5*diffB+y_train_single, weights=None))
        Y_pred_r_val.append(np.average(-diffB+y_train_single))
        Y_pred_check_val.append(np.var(0.5*diffA+0.5*diffB))
        #Y_pred_check_val.append(np.average(np.abs(model.predict([pair_B,x_train_single])+model.predict([x_train_single,pair_B]))))
        Y_median_val.append(np.median(diffA+y_train_single))
        Y_var_val.append(np.var(0.5*diffA-0.5*diffB+y_train_single))
        Y_mse_val.append((Y_pred_val[i]-y_val_single[i])**2)
    
    
    Y_pred_val=cn_transformer.inversetransformY(np.array(Y_pred_val))
    Y_pred_r_val=cn_transformer.inversetransformY(np.array(Y_pred_r_val))
    Y_pred_check_val=np.array(Y_pred_check_val)*(cn_transformer.Ymax)
    Y_median_val=cn_transformer.inversetransformY(np.array(Y_median_val))
    Y_var_val=np.array(Y_var_val)*(cn_transformer.Ymax)**2
    Y_mse_val=np.array(Y_mse_val)*(cn_transformer.Ymax)**2
    Y_self_check_val=np.abs(np.array(model.predict([x_val_single,x_val_single]))).flatten()*(cn_transformer.Ymax)
    
    #####################
    
    Y_pred_test=[]
    Y_pred_r_test=[]
    Y_pred_check_test=[]
    Y_median_test=[]
    Y_var_test=[]
    Y_mse_test=[]
    
    
    
    for i in range(len(x_test_single)):
        pair_B=np.array([x_test_single[i]]*len(x_train_single))
        diffA=model.predict([pair_B,x_train_single]).flatten()
        diffB=model.predict([x_train_single,pair_B]).flatten()
        Y_pred_test.append(np.average(0.5*diffA-0.5*diffB+y_train_single, weights=None))
        Y_pred_r_test.append(np.average(-diffB+y_train_single))
        Y_pred_check_test.append(np.var(0.5*diffA+0.5*diffB))
        #Y_pred_check_test.append(np.average(np.abs(model.predict([pair_B,x_train_single])+model.predict([x_train_single,pair_B]))))
        Y_median_test.append(np.median(diffA+y_train_single))
        Y_var_test.append(np.var(0.5*diffA-0.5*diffB+y_train_single))
        Y_mse_test.append((Y_pred_test[i]-y_test_single[i])**2)
    
    
    Y_pred_test=cn_transformer.inversetransformY(np.array(Y_pred_test))
    Y_pred_r_test=cn_transformer.inversetransformY(np.array(Y_pred_r_test))
    Y_pred_check_test=np.array(Y_pred_check_test)*(cn_transformer.Ymax)
    Y_median_test=cn_transformer.inversetransformY(np.array(Y_median_test))
    Y_var_test=np.array(Y_var_test)*(cn_transformer.Ymax)**2
    Y_mse_test=np.array(Y_mse_test)*(cn_transformer.Ymax)**2
    Y_self_check_test=np.abs(np.array(model.predict([x_test_single,x_test_single]))).flatten()*(cn_transformer.Ymax)
    
    #####################
    
    Y_pred_test2=[]
    Y_pred_r_test2=[]
    Y_pred_check_test2=[]
    Y_median_test2=[]
    Y_var_test2=[]
    Y_mse_test2=[]
    
    
    
    for i in range(len(x_test2_single)):
        pair_B=np.array([x_test2_single[i]]*len(x_train_single))
        diffA=model.predict([pair_B,x_train_single]).flatten()
        diffB=model.predict([x_train_single,pair_B]).flatten()
        Y_pred_test2.append(np.average(0.5*diffA-0.5*diffB+y_train_single, weights=None))
        Y_pred_r_test2.append(np.average(-diffB+y_train_single))
        Y_pred_check_test2.append(np.var(0.5*diffA+0.5*diffB))
        #Y_pred_check_test2.append(np.average(np.abs(model.predict([pair_B,x_train_single])+model.predict([x_train_single,pair_B]))))
        Y_median_test2.append(np.median(diffA+y_train_single))
        Y_var_test2.append(np.var(0.5*diffA-0.5*diffB+y_train_single))
        Y_mse_test2.append((Y_pred_test2[i]-y_test2_single[i])**2)
    
    
    Y_pred_test2=cn_transformer.inversetransformY(np.array(Y_pred_test2))
    Y_pred_r_test2=cn_transformer.inversetransformY(np.array(Y_pred_r_test2))
    Y_pred_check_test2=np.array(Y_pred_check_test2)*(cn_transformer.Ymax)
    Y_median_test2=cn_transformer.inversetransformY(np.array(Y_median_test2))
    Y_var_test2=np.array(Y_var_test2)*(cn_transformer.Ymax)**2
    Y_mse_test2=np.array(Y_mse_test2)*(cn_transformer.Ymax)**2
    Y_self_check_test2=np.abs(np.array(model.predict([x_test2_single,x_test2_single]))).flatten()*(cn_transformer.Ymax)
    
    
    
    
    
    
    print('Train RMSE:', np.average(Y_mse_train)**0.5)
    print('Val RMSE:',np.average(Y_mse_val)**0.5)
    print('Test RMSE:',np.average(Y_mse_test)**0.5)
    print('Test2 RMSE:',np.average(Y_mse_test2)**0.5)    
    
    
    trainrmse=np.average(Y_mse_train)**0.5
    valrmse=np.average(Y_mse_val)**0.5
    testrmse=np.average(Y_mse_test)**0.5
    test2rmse=np.average(Y_mse_test2)**0.5
  
    rmse_train_list.append(trainrmse)
    rmse_val_list.append(valrmse)    
    rmse_test_list.append(testrmse)
    rmse_test2_list.append(test2rmse)
    
    print("prelim mean test rmse",np.mean(rmse_test_list),"+-",np.std(rmse_test_list))
    
    
print("FINAL tnn mean train rmse",np.mean(rmse_train_list),"+-",np.std(rmse_train_list))    
print("FINAL tnn mean val rmse",np.mean(rmse_val_list),"+-",np.std(rmse_val_list))    
print("FINAL tnn mean test rmse",np.mean(rmse_test_list),"+-",np.std(rmse_test_list))
print("FINAL tnn mean test2 rmse",np.mean(rmse_test2_list),"+-",np.std(rmse_test2_list))

end = time.time()
print("time in seconds",end - start)