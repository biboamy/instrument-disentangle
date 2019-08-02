# Music Disentanglement

This repo contains the timbre/pitch disentangled models present in the paper "[Musical Composition Style Transfer via Disentangled Timbre Representations](https://arxiv.org/pdf/1905.13567.pdf)", IJCAI'19.

### Demo
The demo website: https://biboamy.github.io/disentangle_demo/result/

### File structure
#### **disentangled training**: code for training the two models in the paper
- **run.py**: start the training
- **loadData.py**: load the training data
- **model.py**: model structure
- **lib.py**: some untilities for trianing models, including loss function
- **fit.py**: trainer
- **block.py**: model's block structure
- **evl.py**: evaluate the performance during training

#### **IAD classifier**:
- **run.py**: train an IAD classifier

You can choose which model to train by using
`python3 run.py UnetAE_preIP_preRoll`
Details are described in the run.py file.

## Run the code
1. Download the training dataset [Musescore](https://github.com/biboamy/instrument-streaming)
2. Process the data via **process.py**
3. Choose the input model from **run.py**
4. Model will be saved into the directory: **data/model/**

## Extract result
5. For how to get the embedding and the instrument prediction result, please visit 
https://github.com/biboamy/IAD/tree/master/IJCAI'19

## Train an IAD classifier by using embedding
6. IAD classifier/run.py
