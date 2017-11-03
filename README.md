# __Domain Classifier __

___
## Author
* __Qi Hu__ Email: qihuchn@gmail.com

___

## Introduction
 * This is a NLP machine learning project
 * The function of this model is to clasify a single sentence to a certain domain, for example "I' like some Thai food" 
   will be classifid to the RESTAURANT domain.
 * Dataset is not included in this project. You can prepare your own data. The format of data should be as follow:
   1. training, testing and validating data are stored separately in 3 files. Each file is composed of multiple lines 
   (Each line is a data sample).
   2. For each sample of data, there are 3 parts separated by TAB sign ('\t'): domain (label), sentence, 
   word_dictionary_feature (this is a pre-processed feature, representing the high level feature of each word in the sentence)
   
## Algorithm
 * Use word embedding for each raw input word
 * Use CNN model to extract feature from the sentence (after embedding layer)
 * Combine the CNN feature and word_dictionary_feature together and feed into the fully-connected layer
 * Use Softmax for the final classification
 * I tried different feature, parameters and network structures (different position of drop layer and fully-connected layer)

___

## Usage
 * __Train:__ train.py
 * __Test:__ test.py
 * If you want to try different parameters, change them in the train.py/test.py scripts. If you want to tried different 
   models (network structure), change them in corresponding script in the folder 'model'.

## Files
1. __model (folder):__
  Different CNN models. Each file defines one model, and the differences lie in the features and network structure.
2. __utils (folder):__
  Some tool functions for data pre-processing, data loading, error analysis, raw data analysis
3. __train.py:__
  Train the basic model with no word_dictionary_feature
4. __train_vd.py:__
  Train the model with word_dictionary_feature
5. __train_cmp2vd.py:__
  Train the model with simplified word_dictionary_feature (choose only one most important feature for each word)