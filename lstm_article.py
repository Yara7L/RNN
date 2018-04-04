import time
from collections import namedtuple

import numpy as np 
import tensorflow as tf 


with open('train.txt','r') as f:
    text=f.read()
vocab=sorted(set(text))
vocab_to_int={c:i for i,c in enumerate(vocab)}
int_to_vocab=dict(enumerate(vocab))
encoded=np.array([vocab_to_int[c] for c in text],dtype=np.int32)

def get_batches(arr,n_seqs,n_steps):

    characters_per_batch=n_seqs*n_steps
    n_batches=len(arr)//characters_per_batch
    arr=arr[:n_batches*characters_per_batch]

    arr=arr.reshape((n_seqs,-1))

    for n in range(0,arr.shape[1],n_steps):
        x=arr[:,n:n+n_steps]
        y=np.zeros_like(x)
        y[:,:-1],y[:,-1]=x[:,1:],x[:,0]

        yield x,y

