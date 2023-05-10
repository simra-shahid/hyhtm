# +
import os, errno
import time, datetime
import logging as log

class Utils:
    
    def __init__(self, config):
        
        self.config = config
        self.create_output_folder()
        self.create_matrix_folder()
        self.create_experiment_folder()
        
        
    def create_output_folder(self):
        
        path = self.config['result_folder']
        
        try:
            os.makedirs(path)

        except OSError as e:
            if e.errno == errno.EEXIST:
                print("{} Folder to save matrices and results exists".format(path))
           
        
    def create_matrix_folder(self):
        
        path2 = '{}/hyperbolic_{}'.format(self.config['result_folder'], self.config['similarity_matrix']['alpha'])
        
        try: 
            os.mkdir(path1)
            
        except:
            print("Embedding folder exists")
            
        try:
            os.mkdir(path2)
            
        except:
            print("Term-Term Similarity and Hierarchy Matrix folder exists")
        
        
        
    
    def create_experiment_folder(self):
        
        path_to_matrices = '{}/hyperbolic_{}'.format(self.config['result_folder'], self.config['similarity_matrix']['alpha'])
        
        
        expt_name = f"{self.config['result_folder']}/thresh-{self.config['similarity_matrix']['alpha']}_KH-{self.config['hierarchical_matrix']['K']}_KS-{self.config['similarity_matrix']['K']}"
        
        
        try:
            os.makedirs(expt_name)


        except OSError as e:
            if e.errno == errno.EEXIST:
                print("Model is already computed.. Adding timestamp to avoid overwriting")
                timestamp_str = str(datetime.datetime.now())
                print("Timestamp: ", timestamp_str)
                expt_name += timestamp_str
                os.makedirs(expt_name)
                
                
        self.config['result_folder'] = expt_name
        self.config['path_to_matrices'] =  path_to_matrices
    
    
            
    def inputMatrixExists(self): 
         
        hierarchical = False
        clumatrix = False
        
        similarity_path = '{}/input_TFIDF_{}.pt'.format(self.config['path_to_matrices'], self.config['similarity_matrix']['K'])  
        
        if os.path.isfile(similarity_path):
            print("Term-Term Similarity {} exists".format(similarity_path))
            clumatrix = True  

        hierarchical_path = "{}/knn_{}.pt".format(self.config['path_to_matrices'], self.config['hierarchical_matrix']['K'])
        
        if os.path.isfile(hierarchical_path):
            print("Term-Term Hierarchical matrix {} exists".format(hierarchical_path))
            hierarchical = True       
        
        return clumatrix and hierarchical
    
    

    
    
    


    
    
    
    
