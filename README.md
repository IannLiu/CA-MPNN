# CA-MPNN
This is a branch of [chemprop](https://github.com/chemprop/chemprop/tree/v1.7.0) library. This branch intergrates the cross-attention mechanism with D-MPNN architecture to predict the reaction propertirs.  
## Installation  
1. `git clone https://github.com/chemprop/CA-MPNN.git`  
2. `cd CA-MPNN`  
3. `conda env create -f environment.yml`  
4. `conda activate chemprop`  
5. `pip install -e .`
## Training  
To train a cross-attention model, adding `--react_prod_cross_attention` flag to other [chemprop commmands](https://github.com/chemprop/chemprop/blob/v1.7.0/README.md#training).  
## Reaction database
We have extacted [JSF](https://web.stanford.edu/group/haiwanglab/JetSurF/JetSurF2.0/) and [CPL](https://www.sciencedirect.com/science/article/abs/pii/S001021801500317X) reaction databases and mapping the atoms of reactants and products. These databases is openly available on the `reaction_database' file. The details are introucted in our recently peer-revivew papaer
  
