# HyHTM: Hyperbolic Geometry Based Hierarchical Topic Models

This is the code for the paper [Add paper link]

Installation

To set up the environment for the HyHTM code follow the following steps:
    Install all the dependencies from requirements.txt. 
    This version has been tested under Python 3.


Word Embedding used:  Poincare Glove
    Link to embedding files: https://polybox.ethz.ch/index.php/s/TzX6cXGqCX5KvAn
    We use the mix-poincare embeddings: poincare_glove_50x2D_cosh-dist-sq_init_trick
    Place the embeddings in embeddings folder

Place the dataset in datasets folder

To run the code:

python -W ignore codebase/main.py --dataset <dataset> --gpuID $gpuID --alpha <term-term-alpha>  --K_similarity <term-term-similarity K> --K_hierarchy <term-term-hierarchy K> --Ntopics <Ntopics> --levels <levels>


