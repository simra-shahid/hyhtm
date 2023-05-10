import torch, time
import numpy as np
import joblib
import logging as log
from nmf import NMFclass
from sklearn.decomposition import NMF
from collections import Counter, defaultdict
from filtering import * 

torch.cuda.empty_cache()

class modelClass:

    def __init__(self, config, statistical_similarity_matrix, vocabulary, knndists, knninds, hyperbolic_embeddings):
        
        self.config = config
        self.min_number_of_documents = 20 
        
        self.model_output_hierarchy = {}

        self.statistical_similarity_matrix = statistical_similarity_matrix
        self.vocabulary = vocabulary
        self.word2index = {v:k for k, v in dict(enumerate(self.vocabulary)).items()}
        
        self.debug = False
        self.document_level_mapping = defaultdict(list)
        self.cuda = config['cuda'] 

        self.compute_nmf = NMFclass(config)
        self.child_knn_prob, self.child_knn, self.document_bag_of_words = filtering_helper(config['dataset_path'], knndists, knninds, vocabulary)
        


    def get_document_for_topic(self, topics_documents_idx, topic_idx, document_idx):
        '''
        Find documents belonging to this topic
        
        document_idx: [1, 5, 10]
        topics_documents_idx: [0, 1, 0]
        topic_idx: 0
        condition: np.argwhere(topics_documents_idx == topic_idx).squeeze(axis=-1) -> [0, 2]
        document_idx[condition] = [1, 10]
        '''
        condition = np.argwhere(topics_documents_idx == topic_idx).squeeze(axis=-1)
        return document_idx[condition]

    def get_topWords_for_topic(self, topic_word_distribution, top_words=20):

        inds = torch.argsort(topic_word_distribution, descending=True)[:top_words]  # get top words and return the indices in bow representation.
        inds_of_inds = torch.where(topic_word_distribution[inds] != 0)[0]  # keep those indices where prob of word belonging to topic is not 0.
        topWords = inds[inds_of_inds]

        return topWords

    def save_model(self, W, H, parent):
        model_file_path = "{}/nmf_{}.pkl".format(self.config['result_folder'], parent)
        try:
            joblib.dump((W.detach().cpu().numpy(), H.detach().cpu().numpy()), model_file_path)
        except: 
            joblib.dump((W, H), model_file_path)
           
        
        
    def write_topics(self, current_topic, depth, print_words=20):

        output = open('{}/hierarchical_structure.txt'.format(self.config['result_folder']), 'a', encoding="utf-8")
        tabs   = "\t" * depth
        topic  = " ".join(current_topic[:print_words])
        output.write("{}{}\n".format(tabs, topic))
        output.close()

    
    def train(self, current_hierarchy, parent_topic_words=[], document_idx=[], W_parent=[], H_parent=[], current_topic_idx=None):
        
        parent = current_hierarchy.split("_")[-1]
        depth  = int(current_hierarchy.split("_")[-2])
        
        if depth == self.config['structure']['max_depth']: 
            log.info("Maximum depth reached: {}, hierarchy_so_far: {}".format(depth, current_hierarchy))
            #STOP RECURSION FOR THIS BRANCH
            return
            
        if depth!=0 and len(document_idx)<=self.min_number_of_documents:
            logging_text = "L-{}, P-{}".format(depth, parent)
            log.info("Too few documents in {} : {}".format(logging_text, len(document_idx)))
            #STOP RECURSION FOR THIS BRANCH
            return
        
        if len(document_idx)==0:
            num_docs = len(self.statistical_similarity_matrix)
            document_idx = np.arange(len(self.statistical_similarity_matrix))
            self.document_idx = document_idx
            
        else: 
            num_docs = len(document_idx)

        log.info("Exploring P:{}, L:{} with {} documents".format(parent, depth, num_docs))

        if depth==0:
            W, H = self.compute_nmf.run(self.statistical_similarity_matrix, self.config['structure']['N_topics'])

        else:             
            train_matrix   = boost_matrix(self.config['cuda'], W_parent, H_parent, 
                                          self.statistical_similarity_matrix, 
                                          document_idx, current_topic_idx, 
                                          self.document_bag_of_words, 
                                          self.vocabulary, self.word2index, self.child_knn_prob, self.child_knn)
                                        
                               
                
            W, H = self.compute_nmf.run(train_matrix, self.config['structure']['N_topics'])
                  
        if torch.sum(torch.tensor(W).flatten())==0: 
            log.info("Error: W matrix all values are 0")
            return 
        
        if torch.sum(torch.tensor(H).flatten())==0: 
            log.info("Error: H matrix all values are 0")
            return 
        
        self.save_model(W, H, parent)
        
        topics_documents_idx = np.argmax(W.detach().cpu().numpy(), axis=1) 
            
        log.info('')
        log.info('Document split across topics: {}'.format(Counter(topics_documents_idx)))
        log.info('')
             
        for topic_idx, topic_word_distribution in enumerate(torch.tensor(H)):
            
            log.info("Topic: %s | Level: %s | current_hierarchy: %s "%(topic_idx, depth, current_hierarchy))
                
            condition = np.argwhere(topics_documents_idx == topic_idx).squeeze(axis=-1)
            this_topic_documents = document_idx[condition]
            
            current_topic_word_idx = self.get_topWords_for_topic(topic_word_distribution, top_words=100)
            current_topic_words = list(np.asanyarray(self.vocabulary)[current_topic_word_idx.detach().cpu().numpy()])
            self.write_topics(current_topic_words, depth)
            
            self.model_output_hierarchy["L:{}, P:{}, T:{}".format(str(depth), parent, str(topic_idx))] = " ".join(current_topic_words[:20])
            log.info("L:{}, P:{}, T:{}, D:{} | {}".format(str(depth), parent, str(topic_idx), len(this_topic_documents), " ".join(current_topic_words[:20])))
            
            #if len(current_topic_words)<3:
            #    continue
            
            explore_hierarchy = "model_{}_{} {}".format(str(depth + 1), parent, str(topic_idx))
            
            self.train(explore_hierarchy, current_topic_words, this_topic_documents, W, H, topic_idx)
            
            
        del W
        del H

