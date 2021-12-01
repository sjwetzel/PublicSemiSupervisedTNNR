# -*- coding: utf-8 -*-
"""
Created on Fri May  8 20:53:42 2020

@author: anonymous
"""

import numpy as np

class CenterAndNorm:
    
    def __init__(self):
        self.Xmean = 0
        self.Xmax = 0
        self.Ymean = 0
        self.Ymax = 0
    
    def fit(self,X,Y):
        
        self.Xmean = np.mean(X, axis=0)
        self.Xmax = np.max(X - self.Xmean, axis=0)
     
        self.Ymean = 0
        self.Ymax = 1     

    
    def fitX(self,X):
        
        self.Xmean = np.mean(X, axis=0)
        self.Xmax = np.max(X - self.Xmean, axis=0)
    
        
    def transformX(self,X):
        
        X_new=X.copy()
        
        X_new -= self.Xmean
        X_new /= self.Xmax
        
        return X_new
        
    
    def transformY(self,Y):
        
        Y_new=Y.copy()

        Y_new -= self.Ymean
        Y_new /= self.Ymax
        
        return Y_new        
    
    def transform(self,X,Y):
        return self.transformX(X),self.transformY(Y)
    

    def inversetransformX(self,X):
        
        X_new=X.copy()
        
        X_new *= self.Xmax
        X_new += self.Xmean

        return X_new
    
    
    def inversetransformY(self,Y):
        
        Y_new=Y.copy()

        Y_new *= self.Ymax
        Y_new += self.Ymean        
        
        return Y_new
    
    def inversetransform(self,X,Y):
        return self.inversetransformX(X),self.inversetransformY(Y)
    
    def fittransform(self,X,Y):
        self.fit(X,Y)
        return self.transform(X,Y)
    
    def fittransformX(self,X):
        self.fitX(X)
        return self.transformX(X)
    
    
################


def generator(X_data, y_data, batch_size):
    
    X_data=np.array(X_data)
    y_data=np.array(y_data)
    
    num_single_samples = X_data.shape[0]
    batches_per_sweep = num_single_samples/batch_size
    counter=0
    
    perm1=np.random.permutation(len(X_data))
    perm2=np.random.permutation(len(X_data))
  
    while True:

        X1_batch = np.array(X_data[perm1][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        X2_batch = np.array(X_data[perm2][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        y1_batch = np.array(y_data[perm1][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        y2_batch = np.array(y_data[perm2][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        counter += 1
        yield [X1_batch,X2_batch],y1_batch-y2_batch
    
        #restart counter to yield data in the next epoch as well
        if counter >= batches_per_sweep:
            counter = 0

            perm1=np.random.permutation(len(X_data))
            perm2=np.random.permutation(len(X_data))
            
def triple_generator(X_data, y_data, batch_size):
    
    X_data=np.array(X_data)
    y_data=np.array(y_data)
    
    num_single_samples = X_data.shape[0]
    batches_per_sweep = num_single_samples/batch_size
    counter=0
    
    perm1=np.random.permutation(len(X_data))
    perm2=np.random.permutation(len(X_data))
    perm3=np.random.permutation(len(X_data))
  
    while True:

        X1_batch = np.array(X_data[perm1][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        X2_batch = np.array(X_data[perm2][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        X3_batch = np.array(X_data[perm3][batch_size*counter:batch_size*(counter+1)]).astype('float32')        
        
        y1_batch = np.array(y_data[perm1][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        y2_batch = np.array(y_data[perm2][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        y3_batch = np.array(y_data[perm3][batch_size*counter:batch_size*(counter+1)]).astype('float32')

        counter += 1
        yield [X1_batch,X2_batch,X3_batch],np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).transpose()
    
        #restart counter to yield data in the next epoch as well
        if counter >= batches_per_sweep-2:
            counter = 0

            perm1=np.random.permutation(len(X_data))
            perm2=np.random.permutation(len(X_data))  
            perm3=np.random.permutation(len(X_data))

def triple_semi_supervised_generator(X_data, y_data, Unlabeled_data ,batch_size1):
    
    # labeled data for supervised learning
    
    X_data=np.array(X_data)
    y_data=np.array(y_data)
 
    batch_size=batch_size1//2
    
    num_single_samples = X_data.shape[0]
    batches_per_sweep = num_single_samples/batch_size
    counter=0
    
    perm1=np.random.permutation(len(X_data))
    perm2=np.random.permutation(len(X_data))
    perm3=np.random.permutation(len(X_data))
    
    # unlabeled data for semi supervised learning    
    
    Unlabeled_data=np.array(Unlabeled_data)
    Full_data=np.concatenate((X_data,Unlabeled_data),axis=0)   

    num_single_samples_semi = Full_data.shape[0]
    batches_per_sweep_semi = num_single_samples_semi/batch_size
    counter_semi=0
    
    perm1_semi=np.random.permutation(len(Full_data))
    perm2_semi=np.random.permutation(len(Full_data))
    perm3_semi=np.random.permutation(len(Full_data))
  
    while True:

        X1_batch = np.array(X_data[perm1][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        X2_batch = np.array(X_data[perm2][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        X3_batch = np.array(X_data[perm3][batch_size*counter:batch_size*(counter+1)]).astype('float32')        
        
        y1_batch = np.array(y_data[perm1][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        y2_batch = np.array(y_data[perm2][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        y3_batch = np.array(y_data[perm3][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        
        X1_batch_semi = np.array(Full_data[perm1_semi][batch_size*counter_semi:batch_size*(counter_semi+1)]).astype('float32')
        X2_batch_semi = np.array(Full_data[perm2_semi][batch_size*counter_semi:batch_size*(counter_semi+1)]).astype('float32')
        X3_batch_semi = np.array(Full_data[perm3_semi][batch_size*counter_semi:batch_size*(counter_semi+1)]).astype('float32')         
        
        X_A=np.concatenate((X1_batch,X1_batch_semi),axis=0)
        X_B=np.concatenate((X2_batch,X2_batch_semi),axis=0)
        X_C=np.concatenate((X3_batch,X3_batch_semi),axis=0)

        
        y_supervised=np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).astype('float32')   
        y_semi_supervised=np.array([np.zeros(len(X1_batch_semi)),np.zeros(len(X1_batch_semi)),np.zeros(len(X1_batch_semi))]).astype('float32')   
        y_full=np.concatenate((y_supervised.transpose(),y_semi_supervised.transpose()),axis=0)

        counter += 1
        counter_semi +=1
        
        yield [X_A,X_B,X_C],y_full
        #yield [np.concatenate((X1_batch,X2_batch),axis=0),np.concatenate((X2_batch,X2_batch),axis=0),np.concatenate((X3_batch,X3_batch),axis=0)],np.concatenate((np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).transpose(),np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).transpose()),axis=0)
        #yield [X1_batch,X2_batch,X3_batch],np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).transpose()
    
        #restart counter to yield data in the next epoch as well
        if counter >= batches_per_sweep-2:
            counter = 0

            perm1=np.random.permutation(len(X_data))
            perm2=np.random.permutation(len(X_data))  
            perm3=np.random.permutation(len(X_data))
        
        if counter_semi >= batches_per_sweep_semi-2:
            counter_semi = 0

            perm1_semi=np.random.permutation(len(Full_data))
            perm2_semi=np.random.permutation(len(Full_data))
            perm3_semi=np.random.permutation(len(Full_data))



def triple_semi_supervised_generator1(X_data, y_data, Unlabeled_data ,batch_size1):
    
    # labeled data for supervised learning
    
    X_data=np.array(X_data)
    y_data=np.array(y_data)
 
    batch_size=batch_size1//2
    
    num_single_samples = X_data.shape[0]
    batches_per_sweep = num_single_samples/batch_size
    counter=0
    
    perm1=np.random.permutation(len(X_data))
    perm2=np.random.permutation(len(X_data))
    perm3=np.random.permutation(len(X_data))
    
    # unlabeled data for semi supervised learning    
    
    Unlabeled_data=np.array(Unlabeled_data)
    #Full_data=np.concatenate((X_data,Unlabeled_data),axis=0)   

    num_single_samples_semi = Unlabeled_data.shape[0]
    batches_per_sweep_semi = num_single_samples_semi/batch_size
    counter_semi=0
    
    perm1_semi=np.random.permutation(len(Unlabeled_data))
    perm2_semi=np.random.permutation(len(Unlabeled_data))
    perm3_semi=np.random.permutation(len(Unlabeled_data))
  
    while True:

        X1_batch = np.array(X_data[perm1][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        X2_batch = np.array(X_data[perm2][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        X3_batch = np.array(X_data[perm3][batch_size*counter:batch_size*(counter+1)]).astype('float32')        
        
        y1_batch = np.array(y_data[perm1][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        y2_batch = np.array(y_data[perm2][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        y3_batch = np.array(y_data[perm3][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        
        X1_batch_semi = np.array(Unlabeled_data[perm1_semi][batch_size*counter_semi:batch_size*(counter_semi+1)]).astype('float32')
        X2_batch_semi = np.array(Unlabeled_data[perm2_semi][batch_size*counter_semi:batch_size*(counter_semi+1)]).astype('float32')
        X3_batch_semi = np.array(Unlabeled_data[perm3_semi][batch_size*counter_semi:batch_size*(counter_semi+1)]).astype('float32')         
        
        X_A=np.concatenate((X1_batch,X1_batch_semi),axis=0)
        X_B=np.concatenate((X2_batch,X2_batch_semi),axis=0)
        X_C=np.concatenate((X3_batch,X3_batch_semi),axis=0)

        
        y_supervised=np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).astype('float32')   
        y_semi_supervised=np.array([np.zeros(len(X1_batch_semi)),np.zeros(len(X1_batch_semi)),np.zeros(len(X1_batch_semi))]).astype('float32')   
        y_full=np.concatenate((y_supervised.transpose(),y_semi_supervised.transpose()),axis=0)

        counter += 1
        counter_semi +=1
        
        yield [X_A,X_B,X_C],y_full
        #yield [np.concatenate((X1_batch,X2_batch),axis=0),np.concatenate((X2_batch,X2_batch),axis=0),np.concatenate((X3_batch,X3_batch),axis=0)],np.concatenate((np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).transpose(),np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).transpose()),axis=0)
        #yield [X1_batch,X2_batch,X3_batch],np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).transpose()
    
        #restart counter to yield data in the next epoch as well
        if counter >= batches_per_sweep-2:
            counter = 0

            perm1=np.random.permutation(len(X_data))
            perm2=np.random.permutation(len(X_data))  
            perm3=np.random.permutation(len(X_data))
            

        
        if counter_semi >= batches_per_sweep_semi-2:
            counter_semi = 0

            perm1_semi=np.random.permutation(len(Unlabeled_data))
            perm2_semi=np.random.permutation(len(Unlabeled_data))
            perm3_semi=np.random.permutation(len(Unlabeled_data))
            
def triple_semi_supervised_generator2(X_data, y_data, Unlabeled_data ,batch_size1):
    
    # labeled data for supervised learning
    
    X_data=np.array(X_data)
    y_data=np.array(y_data)
 
    batch_size=batch_size1//2
    
    num_single_samples = X_data.shape[0]
    batches_per_sweep = num_single_samples/batch_size
    counter=0
    
    perm1=np.random.permutation(len(X_data))
    perm2=np.random.permutation(len(X_data))
    perm3=np.random.permutation(len(X_data))
    
    # unlabeled data for semi supervised learning    
    
    Unlabeled_data=np.array(Unlabeled_data)
    #Full_data=np.concatenate((X_data,Unlabeled_data),axis=0)   

    num_single_samples_semi = Unlabeled_data.shape[0]
    batches_per_sweep_semi = num_single_samples_semi/batch_size
    counter_semi=0
    
    perm1_semi=np.random.permutation(len(X_data))
    perm2_semi=np.random.permutation(len(Unlabeled_data))
    perm3_semi=np.random.permutation(len(Unlabeled_data))
  
    while True:

        X1_batch = np.array(X_data[perm1][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        X2_batch = np.array(X_data[perm2][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        X3_batch = np.array(X_data[perm3][batch_size*counter:batch_size*(counter+1)]).astype('float32')        
        
        y1_batch = np.array(y_data[perm1][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        y2_batch = np.array(y_data[perm2][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        y3_batch = np.array(y_data[perm3][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        
        X1_batch_semi = np.array(X_data[perm1_semi][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        X2_batch_semi = np.array(Unlabeled_data[perm2_semi][batch_size*counter_semi:batch_size*(counter_semi+1)]).astype('float32')
        X3_batch_semi = np.array(Unlabeled_data[perm3_semi][batch_size*counter_semi:batch_size*(counter_semi+1)]).astype('float32')         
        
        X_A=np.concatenate((X1_batch,X1_batch_semi),axis=0)
        X_B=np.concatenate((X2_batch,X2_batch_semi),axis=0)
        X_C=np.concatenate((X3_batch,X3_batch_semi),axis=0)

        
        y_supervised=np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).astype('float32')   
        y_semi_supervised=np.array([np.zeros(len(X1_batch_semi)),np.zeros(len(X1_batch_semi)),np.zeros(len(X1_batch_semi))]).astype('float32')   
        y_full=np.concatenate((y_supervised.transpose(),y_semi_supervised.transpose()),axis=0)

        counter += 1
        counter_semi +=1
        
        yield [X_A,X_B,X_C],y_full
        #yield [np.concatenate((X1_batch,X2_batch),axis=0),np.concatenate((X2_batch,X2_batch),axis=0),np.concatenate((X3_batch,X3_batch),axis=0)],np.concatenate((np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).transpose(),np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).transpose()),axis=0)
        #yield [X1_batch,X2_batch,X3_batch],np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).transpose()
    
        #restart counter to yield data in the next epoch as well
        if counter >= batches_per_sweep-2:
            counter = 0

            perm1=np.random.permutation(len(X_data))
            perm2=np.random.permutation(len(X_data))  
            perm3=np.random.permutation(len(X_data))
            
            perm1_semi=np.random.permutation(len(X_data))
        
        if counter_semi >= batches_per_sweep_semi-2:
            counter_semi = 0


            perm2_semi=np.random.permutation(len(Unlabeled_data))
            perm3_semi=np.random.permutation(len(Unlabeled_data))
            
            
def triple_semi_supervised_generator3(X_data, y_data, Unlabeled_data ,batch_size1):
    
    # labeled data for supervised learning
    
    X_data=np.array(X_data)
    y_data=np.array(y_data)
 
    batch_size=batch_size1//2
    
    num_single_samples = X_data.shape[0]
    batches_per_sweep = num_single_samples/batch_size
    counter=0
    
    perm1=np.random.permutation(len(X_data))
    perm2=np.random.permutation(len(X_data))
    perm3=np.random.permutation(len(X_data))
    
    # unlabeled data for semi supervised learning    
    
    Unlabeled_data=np.array(Unlabeled_data)
    #Full_data=np.concatenate((X_data,Unlabeled_data),axis=0)   

    num_single_samples_semi = Unlabeled_data.shape[0]
    batches_per_sweep_semi = num_single_samples_semi/batch_size
    counter_semi=0
    
    perm1_semi=np.random.permutation(len(X_data))
    perm2_semi=np.random.permutation(len(X_data))
    perm3_semi=np.random.permutation(len(Unlabeled_data))
  
    while True:

        X1_batch = np.array(X_data[perm1][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        X2_batch = np.array(X_data[perm2][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        X3_batch = np.array(X_data[perm3][batch_size*counter:batch_size*(counter+1)]).astype('float32')        
        
        y1_batch = np.array(y_data[perm1][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        y2_batch = np.array(y_data[perm2][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        y3_batch = np.array(y_data[perm3][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        
        X1_batch_semi = np.array(X_data[perm1_semi][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        X2_batch_semi = np.array(X_data[perm2_semi][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        X3_batch_semi = np.array(Unlabeled_data[perm3_semi][batch_size*counter_semi:batch_size*(counter_semi+1)]).astype('float32')         
        
        X_A=np.concatenate((X1_batch,X1_batch_semi),axis=0)
        X_B=np.concatenate((X2_batch,X2_batch_semi),axis=0)
        X_C=np.concatenate((X3_batch,X3_batch_semi),axis=0)

        
        y_supervised=np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).astype('float32')   
        y_semi_supervised=np.array([np.zeros(len(X1_batch_semi)),np.zeros(len(X1_batch_semi)),np.zeros(len(X1_batch_semi))]).astype('float32')   
        y_full=np.concatenate((y_supervised.transpose(),y_semi_supervised.transpose()),axis=0)

        counter += 1
        counter_semi +=1
        
        yield [X_A,X_B,X_C],y_full
        #yield [np.concatenate((X1_batch,X2_batch),axis=0),np.concatenate((X2_batch,X2_batch),axis=0),np.concatenate((X3_batch,X3_batch),axis=0)],np.concatenate((np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).transpose(),np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).transpose()),axis=0)
        #yield [X1_batch,X2_batch,X3_batch],np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).transpose()
    
        #restart counter to yield data in the next epoch as well
        if counter >= batches_per_sweep-2:
            counter = 0

            perm1=np.random.permutation(len(X_data))
            perm2=np.random.permutation(len(X_data))  
            perm3=np.random.permutation(len(X_data))
            
            perm1_semi=np.random.permutation(len(X_data))
            perm2_semi=np.random.permutation(len(X_data))

        
        if counter_semi >= batches_per_sweep_semi-2:
            counter_semi = 0


            perm3_semi=np.random.permutation(len(Unlabeled_data))
            
def triple_semi_supervised_generator_mix(X_data, y_data, Unlabeled_data ,batch_size1):
    
    # labeled data for supervised learning
    
    X_data=np.array(X_data)
    y_data=np.array(y_data)
 
    batch_size=batch_size1//2
    
    num_single_samples = X_data.shape[0]
    batches_per_sweep = num_single_samples/batch_size
    counter=0
    
    perm1=np.random.permutation(len(X_data))
    perm2=np.random.permutation(len(X_data))
    perm3=np.random.permutation(len(X_data))
    
    # unlabeled data for semi supervised learning    
    
    Unlabeled_data=np.array(Unlabeled_data)
    mixed_data=np.concatenate((X_data,Unlabeled_data),axis=0)   

    num_single_samples_semi = Unlabeled_data.shape[0]
    batches_per_sweep_semi = num_single_samples_semi/batch_size
    counter_semi=0
    
    num_single_samples_mix = mixed_data.shape[0]
    batches_per_sweep_mix = num_single_samples_mix/batch_size
    counter_mix=0
    
    perm1_semi=np.random.permutation(len(X_data))
    perm2_semi=np.random.permutation(len(mixed_data))
    perm3_semi=np.random.permutation(len(Unlabeled_data))
  
    while True:

        X1_batch = np.array(X_data[perm1][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        X2_batch = np.array(X_data[perm2][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        X3_batch = np.array(X_data[perm3][batch_size*counter:batch_size*(counter+1)]).astype('float32')        
        
        y1_batch = np.array(y_data[perm1][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        y2_batch = np.array(y_data[perm2][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        y3_batch = np.array(y_data[perm3][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        
        X1_batch_semi = np.array(X_data[perm1_semi][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        X2_batch_semi = np.array(mixed_data[perm2_semi][batch_size*counter_mix:batch_size*(counter_mix+1)]).astype('float32')
        X3_batch_semi = np.array(Unlabeled_data[perm3_semi][batch_size*counter_semi:batch_size*(counter_semi+1)]).astype('float32')         
        
        X_A=np.concatenate((X1_batch,X1_batch_semi),axis=0)
        X_B=np.concatenate((X2_batch,X2_batch_semi),axis=0)
        X_C=np.concatenate((X3_batch,X3_batch_semi),axis=0)

        
        y_supervised=np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).astype('float32')   
        y_semi_supervised=np.array([np.zeros(len(X1_batch_semi)),np.zeros(len(X1_batch_semi)),np.zeros(len(X1_batch_semi))]).astype('float32')   
        y_full=np.concatenate((y_supervised.transpose(),y_semi_supervised.transpose()),axis=0)

        counter += 1
        counter_semi +=1
        counter_mix +=1
        
        yield [X_A,X_B,X_C],y_full
        #yield [np.concatenate((X1_batch,X2_batch),axis=0),np.concatenate((X2_batch,X2_batch),axis=0),np.concatenate((X3_batch,X3_batch),axis=0)],np.concatenate((np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).transpose(),np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).transpose()),axis=0)
        #yield [X1_batch,X2_batch,X3_batch],np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).transpose()
    
        #restart counter to yield data in the next epoch as well
        if counter >= batches_per_sweep-2:
            counter = 0

            perm1=np.random.permutation(len(X_data))
            perm2=np.random.permutation(len(X_data))  
            perm3=np.random.permutation(len(X_data))
            
            perm1_semi=np.random.permutation(len(X_data))
            
        if counter_mix >= batches_per_sweep_mix-2:
            counter_mix = 0
            perm2_semi=np.random.permutation(len(mixed_data))

        
        if counter_semi >= batches_per_sweep_semi-2:
            counter_semi = 0
            perm3_semi=np.random.permutation(len(Unlabeled_data))   
            
def triple_semi_supervised_generator_all_excluding_labelled_loops(X_data, y_data, Unlabeled_data ,batch_size1):
    
    # labeled data for supervised learning
    
    X_data=np.array(X_data)
    y_data=np.array(y_data)
 
    batch_size=batch_size1//2
    
    num_single_samples = X_data.shape[0]
    batches_per_sweep = num_single_samples/batch_size
    counter=0
    
    perm1=np.random.permutation(len(X_data))
    perm2=np.random.permutation(len(X_data))
    perm3=np.random.permutation(len(X_data))
    
    # unlabeled data for semi supervised learning    
    
    Unlabeled_data=np.array(Unlabeled_data)
    mixed_data=np.concatenate((X_data,Unlabeled_data),axis=0)   

    num_single_samples_semi = Unlabeled_data.shape[0]
    batches_per_sweep_semi = num_single_samples_semi/batch_size
    counter_semi=0
    
    num_single_samples_mix = mixed_data.shape[0]
    batches_per_sweep_mix = num_single_samples_mix/batch_size
    counter_mix=0
    
    perm1_semi=np.random.permutation(len(mixed_data))
    perm2_semi=np.random.permutation(len(mixed_data))
    perm3_semi=np.random.permutation(len(Unlabeled_data))
  
    while True:

        X1_batch = np.array(X_data[perm1][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        X2_batch = np.array(X_data[perm2][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        X3_batch = np.array(X_data[perm3][batch_size*counter:batch_size*(counter+1)]).astype('float32')        
        
        y1_batch = np.array(y_data[perm1][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        y2_batch = np.array(y_data[perm2][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        y3_batch = np.array(y_data[perm3][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        
        X1_batch_semi = np.array(mixed_data[perm1_semi][batch_size*counter_mix:batch_size*(counter_mix+1)]).astype('float32')
        X2_batch_semi = np.array(mixed_data[perm2_semi][batch_size*counter_mix:batch_size*(counter_mix+1)]).astype('float32')
        X3_batch_semi = np.array(Unlabeled_data[perm3_semi][batch_size*counter_semi:batch_size*(counter_semi+1)]).astype('float32')         
        
        X_A=np.concatenate((X1_batch,X1_batch_semi),axis=0)
        X_B=np.concatenate((X2_batch,X2_batch_semi),axis=0)
        X_C=np.concatenate((X3_batch,X3_batch_semi),axis=0)

        
        y_supervised=np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).astype('float32')   
        y_semi_supervised=np.array([np.zeros(len(X1_batch_semi)),np.zeros(len(X1_batch_semi)),np.zeros(len(X1_batch_semi))]).astype('float32')   
        y_full=np.concatenate((y_supervised.transpose(),y_semi_supervised.transpose()),axis=0)

        counter += 1
        counter_semi +=1
        counter_mix +=1
        
        yield [X_A,X_B,X_C],y_full
        #yield [np.concatenate((X1_batch,X2_batch),axis=0),np.concatenate((X2_batch,X2_batch),axis=0),np.concatenate((X3_batch,X3_batch),axis=0)],np.concatenate((np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).transpose(),np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).transpose()),axis=0)
        #yield [X1_batch,X2_batch,X3_batch],np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).transpose()
    
        #restart counter to yield data in the next epoch as well
        if counter >= batches_per_sweep-2:
            counter = 0

            perm1=np.random.permutation(len(X_data))
            perm2=np.random.permutation(len(X_data))  
            perm3=np.random.permutation(len(X_data))
            

            
        if counter_mix >= batches_per_sweep_mix-2:
            counter_mix = 0
            
            perm1_semi=np.random.permutation(len(mixed_data))
            perm2_semi=np.random.permutation(len(mixed_data))

        
        if counter_semi >= batches_per_sweep_semi-2:
            counter_semi = 0

            perm3_semi=np.random.permutation(len(Unlabeled_data))   

            
def triple_semi_supervised_generator_val(X_data, y_data ,batch_size1):
    
    # labeled data for supervised learning
    
    X_data=np.array(X_data)
    y_data=np.array(y_data)
 
    batch_size=batch_size1
    
    num_single_samples = X_data.shape[0]
    batches_per_sweep = num_single_samples/batch_size
    counter=0
    
    perm1=np.random.permutation(len(X_data))
    perm2=np.random.permutation(len(X_data))
    perm3=np.random.permutation(len(X_data))
    
    # unlabeled data for semi supervised learning    
    
    #Full_data=np.concatenate((X_data,Unlabeled_data),axis=0)   


  
    while True:

        X1_batch = np.array(X_data[perm1][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        X2_batch = np.array(X_data[perm2][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        X3_batch = np.array(X_data[perm3][batch_size*counter:batch_size*(counter+1)]).astype('float32')        
        
        y1_batch = np.array(y_data[perm1][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        y2_batch = np.array(y_data[perm2][batch_size*counter:batch_size*(counter+1)]).astype('float32')
        y3_batch = np.array(y_data[perm3][batch_size*counter:batch_size*(counter+1)]).astype('float32')
            
                
        y_supervised=np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).astype('float32')   

        counter += 1
        
        yield [X1_batch,X2_batch,X3_batch],y_supervised.transpose()
        #yield [np.concatenate((X1_batch,X2_batch),axis=0),np.concatenate((X2_batch,X2_batch),axis=0),np.concatenate((X3_batch,X3_batch),axis=0)],np.concatenate((np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).transpose(),np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).transpose()),axis=0)
        #yield [X1_batch,X2_batch,X3_batch],np.array([y1_batch-y2_batch,y2_batch-y3_batch,y3_batch-y1_batch]).transpose()
    
        #restart counter to yield data in the next epoch as well
        if counter >= batches_per_sweep-2:
            counter = 0

            perm1=np.random.permutation(len(X_data))
            perm2=np.random.permutation(len(X_data))  
            perm3=np.random.permutation(len(X_data))
            

# def generator_sym(X_data, y_data, batch_size):
    
#     X_data=np.array(X_data)
#     y_data=np.array(y_data)
    
#     batch_size_sym=batch_size//2
    
#     num_single_samples = X_data.shape[0]
#     batches_per_sweep = num_single_samples/batch_size_sym
#     counter=0
    
#     perm1=np.random.permutation(len(X_data))
#     perm2=np.random.permutation(len(X_data))
  
#     while True:

#         X1_batch = np.array(X_data[perm1][batch_size_sym*counter:batch_size_sym*(counter+1)]).astype('float32')
#         X2_batch = np.array(X_data[perm2][batch_size_sym*counter:batch_size_sym*(counter+1)]).astype('float32')
#         y1_batch = np.array(y_data[perm1][batch_size_sym*counter:batch_size_sym*(counter+1)]).astype('float32')
#         y2_batch = np.array(y_data[perm2][batch_size_sym*counter:batch_size_sym*(counter+1)]).astype('float32')
        
#         X_A=np.concatenate((X1_batch,X2_batch),axis=0)
#         X_B=np.concatenate((X2_batch,X1_batch),axis=0)
        
#         y_A=np.concatenate((y1_batch,y2_batch),axis=0)
#         y_B=np.concatenate((y2_batch,y1_batch),axis=0)
        
#         counter += 1
#         yield [X_A,X_B],y_A-y_B
    
#         #restart counter to yield data in the next epoch as well
#         if counter >= batches_per_sweep:
#             counter = 0

#             perm1=np.random.permutation(len(X_data))
#             perm2=np.random.permutation(len(X_data))
            
            
def generator_sym(X_data, y_data, batch_size):
    
    X_data=np.array(X_data)
    y_data=np.array(y_data)
    
    batch_size_sym=batch_size//2
    
    num_single_samples = X_data.shape[0]
    batches_per_sweep = num_single_samples/batch_size_sym
    counter=0
    counter2=0
    perm1=np.random.permutation(len(X_data))
    perm2=np.random.permutation(len(X_data))
  
    while True:

        X1_batch = np.array(X_data[perm1][batch_size_sym*counter:batch_size_sym*(counter+1)]).astype('float32')
        X2_batch = np.array(X_data[perm2][batch_size_sym*counter:batch_size_sym*(counter+1)]).astype('float32')
        y1_batch = np.array(y_data[perm1][batch_size_sym*counter:batch_size_sym*(counter+1)]).astype('float32')
        y2_batch = np.array(y_data[perm2][batch_size_sym*counter:batch_size_sym*(counter+1)]).astype('float32')
        
        X_A=np.concatenate((X1_batch,X2_batch),axis=0)
        X_B=np.concatenate((X2_batch,X1_batch),axis=0)
        
        y_A=np.concatenate((y1_batch,y2_batch),axis=0)
        y_B=np.concatenate((y2_batch,y1_batch),axis=0)
        
        counter += 1
        yield [X_A,X_B],y_A-y_B
    
        #restart counter to yield data in the next epoch as well
        if counter >= batches_per_sweep:
            counter = 0
            counter2 += 1
            
            perm2=np.roll(perm2,1,axis=0)
            
        if counter2 >= len(X_data):
            counter2 = 0
            
            perm1=np.random.permutation(len(X_data))
            perm2=np.random.permutation(len(X_data))
            

def generator_double(X_1, y_1 ,X_2, y_2, batch_size):
    
    X_1=np.array(X_1)
    y_1=np.array(y_1)
    
    X_2=np.array(X_2)
    y_2=np.array(y_2)
    
    batch_size_sym=batch_size//2
    
    num_pairs = X_1.shape[0]*X_2.shape[0]
    batches_per_sweep = num_pairs/batch_size_sym
    counter=0
    
    indices = np.array([(k,l) for k in range(len(X_1)) for l in range(len(X_2))])
    index_perm = np.random.permutation(len(indices))
    indices = indices[index_perm]
  
    while True:

        X1_batch = np.array(X_1[indices[:,0]][batch_size_sym*counter:batch_size_sym*(counter+1)]).astype('float32')
        X2_batch = np.array(X_2[indices[:,1]][batch_size_sym*counter:batch_size_sym*(counter+1)]).astype('float32')
        y1_batch = np.array(y_1[indices[:,0]][batch_size_sym*counter:batch_size_sym*(counter+1)]).astype('float32')
        y2_batch = np.array(y_2[indices[:,1]][batch_size_sym*counter:batch_size_sym*(counter+1)]).astype('float32')
        
        X_A=np.concatenate((X1_batch,X2_batch),axis=0)
        X_B=np.concatenate((X2_batch,X1_batch),axis=0)
        
        y_A=np.concatenate((y1_batch,y2_batch),axis=0)
        y_B=np.concatenate((y2_batch,y1_batch),axis=0)
        
        counter += 1
        yield [X_A,X_B],y_A-y_B
    
        #restart counter to yield data in the next epoch as well
        if counter >= batches_per_sweep:
            counter = 0
            
            index_perm = np.random.permutation(len(indices))
            indices = indices[index_perm]
            
            
