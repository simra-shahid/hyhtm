import os, numpy as np, pickle
import torch
from tqdm import tqdm
from time import time
from collections import Counter
import logging as log
import json 
import pandas as pd    
from sklearn.feature_extraction.text import CountVectorizer

from term_similarity import GetNeighbourHood, ModifiedTFIDF
from hyperbolic_utils import mix_poincare_distance

    
class PrepareMatrices:
    
    def __init__(self, config, hyperbolic_embeddings): 
        
        self.config = config
        self.hyperbolic_embeddings_model = hyperbolic_embeddings                
        
    def get_vocab_vectors(self, model):
        words_values = []
        sorted_index = list(sorted(self.index_to_word.items(), key=lambda x:x[0]))
        print(sorted_index[:2])
        for (index, word) in sorted_index:
            aux = [word + ' ']
            for k in model[index]:
                aux[0] += str(k) + ' '

            words_values.append(aux[0])
        return words_values
        
        
    def create_vocabulary(self, save_embeddings=True): 
        
                 
        if os.path.exists(self.config['vocab_path']):
            print('Using final vocab')
            with open(self.config['vocab_path'], 'r') as f:
                dataset_vocab = [i.strip("\n").strip(" ") for i in f.readlines()]
                    
                    
        else:
            print('Creating vocab....')
            
            dataset = open(self.config['dataset_path'], 'r')
            cv = CountVectorizer(dataset)
            count_vector=cv.fit_transform(dataset)
            vocab_final = cv.vocabulary_
            dataset_vocab = [k for k, v in vocab_final.items()]
                                        
        hyperbolic_vocab = self.hyperbolic_embeddings_model.vocab
        self.vocabulary = sorted(list(set(dataset_vocab).intersection(hyperbolic_vocab)))
        self.vocabulary_size = len(self.vocabulary)
        
        self.index_to_word = dict(enumerate(self.vocabulary))
        self.word_to_index = {v:k for k,v in self.index_to_word.items()}
        
        print("Vocabulary size", self.vocabulary_size)
        self.hyperbolic_embeddings = dict(zip(list(self.index_to_word.keys()), list(self.hyperbolic_embeddings_model[self.index_to_word.values()])))
        # try: 
            
        # except: 
        #     print("Error Hyperbolic Embeddings is None..")
        #     self.hyperbolic_embeddings = None
        
        if save_embeddings == True:
            
            dimension = 100
            vocab_vectors = self.get_vocab_vectors(self.hyperbolic_embeddings)
            # matrices_path = 'hyperbolic_{}_{}'.format(self.config['similarity_matrix']['K'], self.config['similarity_matrix']['alpha'])
            # print(self.config['result_folder'])
            path = "{}/embeddings.txt".format(self.config['result_folder'])

            with open(path, 'w', encoding="utf-8") as file:
                file.write('{0} {1}\n'.format(self.vocabulary_size, dimension))
                for word_vec in vocab_vectors:
                    file.write("%s\n" % word_vec)
    
    def create_neighbourhood(self):
                
        if int(self.config['similarity_matrix']['K'])> self.vocabulary_size:
            self.config['similarity_matrix']['K'] = self.vocabulary_size
            self.config['hierarchical_matrix']['K'] = self.vocabulary_size
        
        path_to_save = self.config['result_folder']
        
        if(os.path.exists(path_to_save + "/vocab_distances_{}.pt".format(self.config['similarity_matrix']['K']))):
            print("Neighbourhood exists, not recomputing")
            return

        GetNeighbourHood(self.config, self.vocabulary, self.vocabulary_size, self.hyperbolic_embeddings, self.word_to_index, path_to_save)
        print("Neighbourhood created")
        
    def create_TFIDF(self):
                
        path = self.config['result_folder']

        if os.path.exists(path + "/input_TFIDF_{}.pt".format(self.config['similarity_matrix']['K'])):
            print("Input TFIDF exists, not recomputing")
            return
        
        
        TFIDF = ModifiedTFIDF(cuda = self.config['cuda'], 
                                 dataset_file_path = self.config['dataset_path'], 
                                 n_words = self.vocabulary_size,
                                 path_to_save = path,
                                 K_S = self.config['similarity_matrix']['K'])

        input_TFIDF = TFIDF.fit_transform()

        torch.save({"tfidf": input_TFIDF, "feature_names":TFIDF.vocab}, '{}/input_TFIDF_{}.pt'.format(path, self.config['similarity_matrix']['K']))
        
        del input_TFIDF
        
        
    def get_hyperbolic_nearest_neighbours(self):
        
        word_inds = list(self.hyperbolic_embeddings.keys())
  
        candidate_list = [self.hyperbolic_embeddings[x] for x in word_inds]
        candidates = torch.from_numpy(np.asarray(candidate_list).astype(np.float32)).to(self.config['cuda'])
        
        dists_all = torch.zeros((self.vocabulary_size, self.config['hierarchical_matrix']['K']))
        indices   = torch.zeros((self.vocabulary_size, self.config['hierarchical_matrix']['K']),dtype=int)
        
        for i in tqdm(range(self.vocabulary_size)):
            input_ = candidates[i]
            dists = mix_poincare_distance(input_, candidates)
            
            knn_indexes = torch.argsort(dists)[:self.config['hierarchical_matrix']['K']]
            indices[i]   = knn_indexes
            dists_all[i] = dists[knn_indexes]
                                      
            del knn_indexes, dists

        candidates.to('cpu')
        del candidates
        return dists_all, indices

    def create_hierarchical_matrix(self):
        
        reference_path = "{}/knndists_{}.pt".format(self.config['result_folder'], self.config['hierarchical_matrix']['K'])
        
        if(os.path.exists(reference_path)):
            print("Knn distances exists") 
            
        else:
            distances, indices = self.get_hyperbolic_nearest_neighbours()
            torch.save({'dists':distances,'inds':indices}, reference_path)
            
                
    def run(self): 
        
        self.create_vocabulary()
        self.create_neighbourhood()
        self.create_TFIDF() 
        self.create_hierarchical_matrix()
        
        
        

