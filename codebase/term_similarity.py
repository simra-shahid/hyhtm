import timeit
import warnings
import torch
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from track_memory import track
from tqdm import tqdm
from hyperbolic_utils import mix_poincare_distance

        

class GetNeighbourHood:
    
    def __init__(self, config, vocabulary, n_words, word_matrix, word_to_index, path_to_save):

        self.config = config
        self.n_words = n_words
        self.path_to_save = path_to_save
        
        self.words = vocabulary 
        self.word_vectors = np.array([word_matrix[word_to_index[word]] for word in self.words])
        
        self.create_mix_poincare()
            
            
    def get_knn(self):

        word_vectors = torch.from_numpy(self.word_vectors.astype(np.float32))
        dists_all = torch.zeros((self.n_words, self.config['similarity_matrix']['K']))
        indices   = torch.zeros((self.n_words, self.config['similarity_matrix']['K']),dtype=int)
        for i in tqdm(range(self.n_words)):
            dists = mix_poincare_distance(word_vectors[i], word_vectors)
            knn_indexes = torch.argsort(dists)[:self.config['similarity_matrix']['K']]
            indices[i]  = knn_indexes
            dists_all[i]= dists[knn_indexes]
            del dists
            del knn_indexes

        return dists_all, indices

    def create_mix_poincare(self):
        self.distances, self.indices = self.get_knn()
        torch.save({'dists':self.distances,'inds':self.indices},"{}/knndists_{}.pt".format(self.path_to_save, self.config['similarity_matrix']['K']))
        self._save()

    def _save(self):
        
        distances = self.distances.to('cpu')
        indices   = self.indices.to('cpu')
        
        list_words = torch.zeros((self.n_words, self.n_words), dtype=torch.float32)
        
        largest = torch.max(distances)
        for i in range(self.n_words):
            list_words[i][indices[i]] = 1-distances[i]/largest
        
        torch.save({"data":list_words, "index":self.words, "vocab":self.words},'{}/vocab_distances_{}.pt'.format(self.path_to_save, self.config['similarity_matrix']['K']))
        
        del list_words
        del self.distances
        del self.indices
        del distances
        del indices




class ModifiedTFIDF:
    """
    Calculates Term Frequency-Inverse Document Frequency (TFIDF) using equation 5 in HyHTM paper. 

    """

    def __init__(self, cuda, dataset_file_path, n_words, path_to_save, K_S):
        self.cuda = cuda
        # self.cuda = "cuda:{}".format(cuda) if cuda!=-1 else "cpu"
        self.dataset_file_path = dataset_file_path
        self.n_words = n_words

        loaded = torch.load('{}/vocab_distances_{}.pt'.format(path_to_save, K_S))
        self.vocab = loaded['vocab']
        self.vocab_data = loaded['data']

        del loaded
        self._read_input()

    def _read_input(self):
        arq = open(self.dataset_file_path, 'r', encoding="utf-8")
        
        doc = arq.readlines()
        arq.close()
            
        self.documents = list(map(str.rstrip, doc))
        self.n_documents = len(self.documents)

    def fit_transform(self):

        print('\nComputing TF...')
        self.compute_tf()
        
        print('\nComputing IDF...')
        self.compute_idf()
        print()
        
        similarity_tfIDF = torch.mul(self.tf.to(self.cuda), self.idf.reshape(1,                          self.idf.shape[0]).to(self.cuda))
        
        if self.cuda!='cpu':
            similarity_tfIDF.to('cpu')
            self.idf.to('cpu')
            self.tf.to('cpu')

        return similarity_tfIDF
    
    def _raw_tf(self, binary=False):
        tf_vectorizer = CountVectorizer(input=self.documents, max_features=self.n_words, binary=binary, vocabulary=self.vocab)
        tf = tf_vectorizer.fit_transform(self.documents)
        return torch.from_numpy(tf.toarray().astype(np.float32))

    def compute_tf(self):
        

        tf = self._raw_tf()
        
        self.tf = torch.zeros(self.n_documents, len(self.vocab))
        inds = torch.tensor(list(range(0, len(self.vocab))))
        self.hyp_aux = self.vocab_data[inds].type(torch.float32).to(self.cuda)
        self.tf = torch.mm(tf.to(self.cuda), torch.transpose(self.hyp_aux,0,1))
        
        if self.cuda!='cpu':
            self.tf.to("cpu")
            self.hyp_aux.to("cpu")
            # tf.to("cpu")
        


    def compute_idf(self):

        
        
        out = torch.empty((self.tf.shape[0], self.hyp_aux.shape[1]), dtype=torch.float32).to(self.cuda)
        _dot = torch.matmul(self.tf, torch.transpose(self.hyp_aux, 0, 1), out=out)

        if self.cuda!='cpu':
            self.hyp_aux.to("cpu")
            _dot.to("cpu")
            self.tf.to("cpu")
            out.to("cpu")
        
        self.hyp_aux.to(self.cuda)
        bin_hyp_aux = torch.nan_to_num(torch.divide(self.hyp_aux, self.hyp_aux))
        
        if self.cuda!='cpu': 
            self.hyp_aux.to("cpu")

        track()

        self.tf.to(self.cuda)
        bin_hyp_aux = torch.transpose(bin_hyp_aux, 0, 1)
        y2 = bin_hyp_aux.shape[1]
        y1 = self.tf.shape[0]
        out1 = torch.empty((y1, y2), dtype=torch.float32).to(self.cuda)
        _dot_bin = torch.matmul(self.tf, bin_hyp_aux,  out=out1)
        
        if self.cuda!='cpu':
            _dot_bin.to("cpu")
            out1.to("cpu")
            self.tf.to("cpu")
            bin_hyp_aux.to("cpu")


        mu_hyp = torch.nan_to_num(torch.divide(_dot.to(self.cuda), _dot_bin.to(self.cuda)))

        self.idf = torch.sum(mu_hyp, dim=0)
        self.idf = torch.log10(torch.divide(torch.tensor(self.n_documents), self.idf))
        
        if self.cuda!='cpu':
            self.idf.to('cpu')
            mu_hyp.to("cpu")
            _dot.to("cpu")
            _dot_bin.to("cpu")
