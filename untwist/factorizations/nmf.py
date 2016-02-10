"""
Non-negative Matrix Factorization algorithms.
Many can be unified under a single framework based on multiplicative updates.
Implementation is inspired by NMFlib by Graham Grindlay
"""
import numpy as np
from sklearn.preprocessing import normalize

from untwist.base.algorithms import Processor
from untwist.data import audio


class NMF(Processor):
    
    def __init__(self, rank, update_func = "kl", iterations = 100, 
        threshold = None, norm_W = 2, norm_H = 0, 
        update_W = True, update_H = True):

        self.rank = rank
        self.update = getattr(self, update_func + "_updates")
        self.iterations = iterations
        self.threshold = threshold
        self.norm_W = norm_W
        self.norm_H = norm_H
        self.update_W = update_W
        self.update_H = update_H

    """
    Compute divergence between reconstruction and original
    """    
    def compute_error(self, V, W, H):
        Vf = V.flatten()
        Rf = np.dot(W, H).flatten()
        err = np.sum(np.multiply(Vf, np.log(Vf/Rf)) - Vf + Rf)
        return err        

    """
    Normalize W and/or H depending on initialization options
    """    
    def normalize_W(self, W):        
        if self.norm_W > 1:            
            W = normalize(W, norm = 'l'+str(self.norm_W), axis = 0)
        return W
        
    def normalize_H(self, H):
        if self.norm_H > 1:
            H = normalize(H, norm = 'l'+str(self.norm_H), axis = 1)
        return H

    def normalize(self, W, H):
        W = self.normalize_W(W)
        H = self.normalize_H(H)
        return [W,H]

    """
    Initialize and compute multiplicative updates iterations
    """                
    def process(self, V, W0 = None, H0 = None):
         W = W0 if W0 is not None else np.random.rand(V.shape[0],self.rank)
         H = H0 if H0 is not None else np.random.rand(self.rank, V.shape[1])
         self.ones = np.ones(V.shape)
         [W, H] = self.normalize(W, H)
         for i in range(self.iterations):
             [V, W, H] = self.update(V, W,H)
             if self.threshold is not None:
                 err = self.compute_error(V, W, H)
                 if err <= self.threshold:
                     return [W, H, err]
             print i
         return [W, H]

    """
    Multiplicative updates functions
    """    
         
    """
    Optimize Kullback-Leibler divergence
    """    
    def kl_updates(self, V, W, H):
        eps = np.spacing(1)
        if self.update_W:            
            W  = W * np.dot(V / (np.dot(W, H) + eps) , H.T) / np.max(np.dot(self.ones, H.T), eps)
            np.nan_to_num(W)
        self.normalize_W(W)        
        if self.update_H:             
            H  = H * np.dot(W.T, V / (np.dot(W, H )+ eps)) / np.max(np.dot(W.T, self.ones), eps)
            np.nan_to_num(H)
        self.normalize_H(H)

        return [V, W, H]
