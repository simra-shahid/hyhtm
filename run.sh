alpha=0.1
K_hierarchy=500
K_similarity=500
Ntopics=10
levels=3
dataset=nips
gpuID=1

python -W ignore codebase/main.py --dataset $dataset --gpuID $gpuID --alpha $alpha  --K_similarity $K_similarity --K_hierarchy $K_hierarchy --Ntopics $Ntopics --levels $levels 