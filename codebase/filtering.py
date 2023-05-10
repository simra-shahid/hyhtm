from collections import defaultdict
import pickle 
import torch
from gensim.models.keyedvectors import KeyedVectors
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from scipy.special import softmax



try:
    f = open('stopwords.txt', 'r')
except:
    f = open('../stopwords.txt', 'r')
stopwords = f.readlines()
stopwords = [i.strip("\n") for i in stopwords]



def filtering_helper(dataset_path, knnmatrix, knninds, vocabulary):
    
    actual_knn = np.zeros((len(vocabulary), len(vocabulary)))

    knnmatrix = knnmatrix.cpu().clone().detach()
    knninds = knninds.cpu().clone().detach()
    i = 0
    for row in knninds:
        actual_knn[i,row] = 1-softmax(knnmatrix[i])
        i+=1
            
    child_matrix_probs = np.where(np.transpose(actual_knn)>0, np.transpose(actual_knn), 0)
    child_matrix = np.where(np.transpose(actual_knn)>0, 1, 0)

    documents = open(dataset_path, 'r').readlines()
    tf_vectorizer = CountVectorizer(max_features=len(vocabulary), vocabulary=vocabulary)
    tf = tf_vectorizer.fit_transform(documents)
    document_bag_of_words = tf.toarray()

    return child_matrix_probs, child_matrix, document_bag_of_words



def parent_children_filtering(topic_words, vocab, word2ind, actual_knn, restrict_each_topic_word = 2000, topic_words_to_consider = 30):
    
    #topic_words = topic_words[:topic_words_to_consider]
    child_matrix = np.zeros((len(vocab), len(vocab)))
    
    for word in topic_words:
        parent = word2ind[word]
        nonzero_inds = np.argwhere(actual_knn[:,parent]>0)
        inds_to_leave = nonzero_inds.shape[1]
        child_inds = np.argsort(actual_knn[:,parent].flatten())[-inds_to_leave:][:restrict_each_topic_word]
        child_matrix[parent] = np.array([1 if i in child_inds or i==parent else 0 for i in range(len(vocab)) ])
                              
    return child_matrix

                                

def boost_matrix(cuda, W_parent, H_parent, inputTFIDF, 
                 document_idx, this_topic_idx, document_bag_of_words, 
                 vocab, word2ind, child_knn_prob, child_knn, topic_term_threshold = 0.5): 


    # Step1: From H_parent, get the topic term probability of current topic to explore
    topic_term_probabilities = np.array(H_parent.cpu().clone().detach())[this_topic_idx]
    
    
    # Step 2: Create Parent - Child Matrix 

    topic_term_filter = np.where(topic_term_probabilities >= topic_term_threshold, topic_term_probabilities, 0)
    first_matrix = np.multiply(document_bag_of_words[document_idx], topic_term_filter)
        
    child_matrix = child_knn
    
    # Step 4: Multiply first and second matrix 
    
    first_matrix = torch.tensor(first_matrix, dtype=torch.float32)
    first_matrix.to(cuda)
    second = torch.tensor(child_matrix, dtype=torch.float32)
    second.to(cuda)
    
    
    intermediate = torch.matmul(first_matrix, second).to(cuda)
    first_matrix.to('cpu')
    second.to('cpu')
    

    # Step 5: Multiply intermediate output with clumatrix
    subset_documents_TFIDF = inputTFIDF[document_idx]
    subset_documents_TFIDF.to(cuda)
    
    output_matrix = torch.multiply(subset_documents_TFIDF, intermediate).to(cuda)
     
    subset_documents_TFIDF.to('cpu')
    intermediate.to('cpu')
    
    output_matrix = output_matrix/torch.linalg.norm(output_matrix)
    output_matrix.to('cpu')
    
    
    return output_matrix 
                            
                             
    
