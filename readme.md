# HGNN-S

Code and dataset for the paper ...

## SciBERT Pre-Trained Models

- [**SciBERT**](https://github.com/allenai/scibert/) ([arXiv](https://arxiv.org/pdf/1903.10676.pdf)) InfoXLM: An Information-Theoretic Framework for Cross-Lingual Language Model Pre-Training.

## Dependencies
- six==1.15.0
- numpy==1.19.2
- tqdm==4.49.0
- transformers==3.2.0
- nltk==3.5
- matplotlib==3.3.3
- torch==1.6.0
- scikit_learn==0.24.1

## How To Use

### Pre-Trained Models for Stage

In the paper, we used the pre-trained model provided by [**SciBERT**](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_uncased.tar)

### Training HGNN-s Models

#### Preparing Training Data

**Academic-II(A-II) data** 

In the paper, we use the Academic-II(A-II) [Zhang et al., 2019]. as the training data. You can get the training data by xxx.

**Aminer-AND data** 

The dataset is released by [Zhang et al.,2018], which contains 199,941 papers, 403,441 authors and 11,716 venues. we use it as the test data. You can get the training data by xxx.


#### Evaluating
```
python HGNN-S_evaluation.py
--data_path ../../data/test                                             # test data location
```