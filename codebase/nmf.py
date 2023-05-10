# +
import torch, time
import numpy as np
import joblib
import logging as log
from sklearn.decomposition import NMF
from sklearn.utils import check_random_state



class NMFclass:
    def __init__(self, config):
        self.config = config

    
    def initialise_WH(self, X, n_components, random_state, H_init=None): 
        
        avg = np.sqrt(X.mean() / n_components)
        rng = check_random_state(random_state)

        W_init = avg * rng.standard_normal(size=(X.shape[0], n_components)).astype('double', copy=False)
        W_init = np.abs(W_init)
            
        return W_init, H_init 

    
    def p_onmf(self, X, rank, H_init=None, W_init=None, iterations=200, alpha=1.0):
        
        
        m, n = X.shape
        W = torch.rand(m, rank).to(self.config['cuda']) if isinstance(W_init, type(None)) else W_init
        H = torch.rand(rank, n).to(self.config['cuda']) if isinstance(H_init, type(None)) else H_init
        
        for itr in range(iterations):
            
            enum = torch.mm(X, torch.transpose(H, 0, 1))
            denom = torch.mm(W, torch.mm(H, torch.transpose(H, 0, 1)))
            W = torch.nan_to_num(torch.mul(W, torch.div(enum, denom)))
            
            HHTH = torch.mm(torch.mm(H, torch.transpose(H, 0, 1)), H)
            enum = torch.mm(torch.transpose(W, 0, 1), X) + torch.mul(H, alpha)
            denom = torch.mm(torch.mm(torch.transpose(W, 0, 1), W), H) + torch.mul(HHTH, 2.0 * alpha)
            H = torch.nan_to_num(torch.mul(H, torch.div(enum, denom)))
        
        W.to('cpu') 
        H.to('cpu') 
        
        return W, H
    
   

    def run(self, X, rank, H_init=None, W_init=None):
        
        X = X.to(self.config['cuda'])
        W, H = self.p_onmf(X, rank, H_init)

        X.to('cpu')
        del X
        return W, H
