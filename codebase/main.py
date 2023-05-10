import time, datetime, argparse, random, os, pickle, pprint
import logging as log 
import torch
from gensim.models.keyedvectors import KeyedVectors
from collections import Counter
import linecache
import os
from training import TrainingPipeline
from prepare_matrices import PrepareMatrices
from utils import Utils
from hyperbolic_utils import *


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", action="store", dest="dataset",
                      help="Directory to find matrices for the dataset", default=None)
    
    parser.add_argument("--alpha", action="store", dest="alpha", type=float,
                      help="Threshold used for setting the threshold in Equation 4", default=0.4)

    parser.add_argument("--K_similarity", action="store", type=int, dest="similarity_matrix_K",
                      help="Term-Term Similarity k_s in Equation 4", default=500)
    
    parser.add_argument("--K_hierarchy", action="store", type=int, dest="hierarchical_matrix_K",
                      help="Term-Term Hierarchy k_h in Equation 4", default=100)
        
    parser.add_argument("--Ntopics", action="store", type=int, dest="Ntopics", 
                        help="Number of Topics", default=10)
    
    parser.add_argument("--levels", action="store", type=int, dest="levels",
                      help="Hierarchy Depth", default=3)
    
    parser.add_argument("--gpuID", action="store", dest="gpuID", 
                        help="gpuID", default="-1")

    args = parser.parse_args()
   
    if args.gpuID == "-1": 
        cuda = "cpu"
    else:
        cuda = "cuda:0"
    
    CODEBASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    config = {
        
        'dataset': args.dataset, 
        'result_folder' : '{}_hyhtm'.format(args.dataset), 
        'cuda': cuda, 
        
        #Preparing matrices
        'hyperbolic_embeddings_path': "{}/embeddings/poincare_glove_50x2D_cosh-dist-sq_init_trick".format(CODEBASE_PATH),
        'hyperbolic_dim': 100,
        
        'dataset_path' : "{}/datasets/{}/dataset_{}.txt".format(CODEBASE_PATH, args.dataset, args.dataset), 
        'vocab_path' : "{}/datasets/{}/vocab.txt".format(CODEBASE_PATH, args.dataset, args.dataset),
        
        # Structure of hierarchy
        'structure':{ 
            'N_topics' : args.Ntopics,
            'max_depth' : args.levels},
                
        # Metric Space
        'similarity_matrix': {
                'alpha' : args.alpha,
                'K': args.similarity_matrix_K,
            },

        # For term-term hierarchical matrix
        'hierarchical_matrix': {
                'distance_function': "knn",
                'K': args.hierarchical_matrix_K
            }    
    }
    

    
    hyperbolic_embeddings = KeyedVectors.load_word2vec_format(config['hyperbolic_embeddings_path'], binary=False)                 
    verbose = True
    
    utils_obj = Utils(config)

    # config = utils_obj.config
    
    for handler in log.root.handlers[:]:
        log.root.removeHandler(handler)
        
    log_file_name = '{}/hyhtm.log'.format('{}/{}'.format(CODEBASE_PATH, config['result_folder'])) 
    log.basicConfig(filename=log_file_name, 
                    format='%(asctime)-18s %(levelname)-10s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d/%m/%Y %H:%M', 
                    level=log.INFO)

    log.info("Config parameters: {}".format(config))
    
    if verbose: 
        print("Config parameters: ", end = " ") 
        pprint.pprint(config)
        
    if verbose: 
        print("Loading matrices and embeddings...")
        
    log.info("Loading matrices and embeddings...")

    model = TrainingPipeline(config)

    if not utils_obj.inputMatrixExists():
        log.info("Term-Term similarity and hierarchy matrix does not exist.")
        if verbose: print("Term-Term similarity and hierarchy matrix does not exist.")
        pp = PrepareMatrices(config, hyperbolic_embeddings)
        pp.run() 

    else: 
        if verbose: print("Term-Term similarity and hierarchy matrix exists")
        log.info("Term-Term similarity and hierarchy matrix exists")
    
    log.info("Generating topics...")
    if verbose: print('Generating topics...')

    training_time, hierarchy = model.generate_topics(hyperbolic_embeddings)    
    if verbose: pprint.pprint(hierarchy)

if __name__ == '__main__':
    main()


