import logging as log
import time
import torch
from model import modelClass
import pickle 
import numpy as np 
import os 


class TrainingPipeline:

    def __init__(self, config):
        self.config = config

    def load_precomputed_matrix(self):
        
        sim_path = '{}/input_TFIDF_{}.pt'.format(self.config['result_folder'], self.config['similarity_matrix']['K'])
        statistical_similarity_matrix = torch.load(sim_path)
        vocabulary = list(statistical_similarity_matrix['feature_names'])
        statistical_similarity_matrix = statistical_similarity_matrix['tfidf']
        
        knninstance = "{}/knndists_{}.pt".format(self.config['result_folder'], self.config['hierarchical_matrix']['K'])
        
        
        knninstance = torch.load(open(knninstance, 'rb'))
        knndists = knninstance["dists"]
        knninds  = knninstance["inds"]
        
        return vocabulary, statistical_similarity_matrix, knndists, knninds


    def generate_topics(self, hyperbolic_embeddings):
        
        vocabulary, statistical_similarity_matrix, knndists, knninds = self.load_precomputed_matrix()

        full_start = time.time()
        current_hierarchy = "model_{depth}_{parent_topic}".format(depth=0, parent_topic='-1')
        model = modelClass(self.config, statistical_similarity_matrix,  vocabulary, knndists, knninds, hyperbolic_embeddings)
        model.train(current_hierarchy)
        
        return time.time()-full_start, model.model_output_hierarchy
    
   

