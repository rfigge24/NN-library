# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 21:17:07 2022

@author: rfigg
"""
import losses
import numpy as np

class Neural_Network:
    def __init__(self, 
                 layers, 
                 loss_function=losses.MSE, 
                 solver='SGD'):                                                
        #
        self.layers = layers
        self.loss_function = loss_function
        self.solver=solver  
 #   
    def predict(self,input_data):
        output = input_data  
        for layer in self.layers:
            output = layer.forward_propagation(output)
    
        return output       
 #
 
    def fit(self, 
            X_train, 
            y_train, 
            epochs = 1, 
            learning_rate = 0.001,
            batch_size=200,                                                    
            shuffle=True,                                                      
            verbose = True):
        
        
        #check for shape similarity of the training sets:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError('X_train and y_train should have the same amount of samples!!')
        #check for correct shapes of training sets:
        if len(X_train.shape) != 2 and len(y_train.shape) != 2:
            raise ValueError('Both X_train and y_train should be 2d np arrays with shape nxd, n(samples) and d(dimensions per sample)!')
        
        
        #correcting the batch_size:
        batch_size = (1 if self.solver =='SGD' 
                      else batch_size if batch_size < X_train.shape[0] 
                      else X_train.shape[0])
        
        #training algoritm that gets looped epoch times:
        for e in range(epochs):
            error = 0
            
            #preparing training data:
                #shuffle the trainingsets if shuffle is True:
            if shuffle:
                rng = np.random.default_rng()
                shuffled_indices = rng.permutation(X_train.shape[0])
                X_train = X_train[shuffled_indices]
                y_train = y_train[shuffled_indices]
                
                #make a list of batches by splitting the datasets:
            batches = list(zip(np.array_split(X_train,batch_size),
                          np.array_split(y_train,batch_size)))
            
            #loop the train algoritm for all batches
            for X_batch, y_batch in batches:
                #forward:
                output = self.predict(X_batch)
                #error:
                error += self.loss_function.loss(y_batch, output)
                #backward:
                grad = self.loss_function.loss_prime(y_batch, output) 
                for layer in reversed(self.layers):
                    grad = layer.backward_propagation(grad, learning_rate)
            error /= len(batches)

            #verbose printing part:
            if verbose:
                print('epoch: %d/%d, error=%.8f' %(e+1, epochs, error))            
        self.error = error
        
