# HyHTM: Hyperbolic Geometry Based Hierarchical Topic Models

HyHTM is an implementation of a hierarchical topic modeling algorithm using hyperbolic geometry. 
The algorithm is described in detail in the paper [Add paper link].

#### Installation
To set up the environment for the HyHTM code, please follow these steps:
 - Install all the dependencies from requirements.txt. This version has been tested under Python 3.
 - Download the word embeddings used in the paper, which are Poincare Glove embeddings. The embeddings can be found at this link: https://polybox.ethz.ch/index.php/s/TzX6cXGqCX5KvAn. We use the mix-poincare embeddings called "poincare_glove_50x2D_cosh-dist-sq_init_trick". Place the embeddings in the "embeddings" folder.
 - Place your dataset in the "datasets" folder.

#### Running the code
To run the HyHTM code, use the following command in your terminal:

    python -W ignore codebase/main.py --dataset <dataset> --gpuID $gpuID --alpha <term-term-alpha> --K_similarity <term-term-similarity K> --K_hierarchy <term-term-hierarchy K> --Ntopics <Ntopics> --levels <levels>

#### Here are the descriptions of the parameters:

--dataset: the name of the dataset you want to use.
--gpuID: the ID of the GPU you want to use. If you want to use the CPU, use -1.
--alpha: the value of the term-term alpha parameter.
--K_similarity: the number of dimensions used to approximate the similarity matrix.
--K_hierarchy: the number of dimensions used to approximate the hierarchy matrix.
--Ntopics: the number of topics you want to discover.
--levels: the number of levels you want to use in the hierarchical model.

Note that the --gpuID parameter is optional. If you don't provide it, the code will use the CPU by default.


Please cite our paper if you use this code for your research.