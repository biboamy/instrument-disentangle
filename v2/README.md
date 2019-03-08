# Instrument streaming

This repo contains the timbre/pitch disentangled models present in the paper "Learning Disentangled Representations for
Timber and Pitch in Music Audio"

### Demo
The demo website: [coming soon]

### File structure
- function: all the necessary code for running the model
- data: put the training data
- run.py: choose the model and run the training
- model.py: define the model structure
- block.py: define the detailed structure
- fit.py: start the training 
- loadData.py: load training data
- lib: loss function, normalization function, split batch function

## Run the code
1. Download the training data from [url]
2. Choose the input model from **run.py**
3. Model will be saved into the directory: **data/model/**
