import numpy as np
import pandas as pd
import os
print os.getcwd()
class Data_Reader():
    def __init__(self,dataset,datatype, batch_size =None):
        npz = np.load('/home/atharva/Notebooks/{}_data_{}.npz'.format(dataset,datatype))
        self.inputs, self.targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)
        
        if batch_size is None:
            self.batch_size = self.inputs.shape[0]
        else:
            self.batch_size = batch_size
        
        self.curr_batch = 0
        self.batch_count = self.inputs.shape[0] // self.batch_size


    def next(self):
        if self.curr_batch >=self.batch_count:
            self.curr_batch = 0
            raise StopIteration()            
        
        
        # We slice the dataset in batches and then the "next" function loads them one after another
        
        batch_slice = slice(self.curr_batch, (self.curr_batch +1)* self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self.curr_batch += 1
        
        # The function will return the inputs batch and the one hot encoded targets.
        return inputs_batch, targets_batch
    
    def __iter__(self):
        return self
    
