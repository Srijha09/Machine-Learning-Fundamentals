# -*- coding: utf-8 -*-
#function for tokenizing the corpus
def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens

import torch
from torch.autograd import Variable
import torch.functional as F
import torch.nn.functional as F
#importing variables from main.py
from  main1 import corpus
from main1 import tokenized_corpus
import numpy as np




from collections import Counter
def get_word_frequencies(tokenized_corpus):
  frequencies = Counter()
  for sentence in corpus:
    for word in sentence:
      frequencies[word] += 1
  freq = frequencies.most_common()
  return freq


# let mmsg= context and let fde=centre-word
#We can now generate pairs fde and mmsg. Let’s assume context window to be symmetric and equal to 2.
from main1 import word2idx,pair_size  
def wordvec():
    window_size = 2
    idx_pairs = []
# for each sentence
    for sentence in tokenized_corpus:
        indices = [word2idx[word] for word in sentence]
    # for each word, threated as center word
        for center_word_pos in range(len(indices)):
        # for each window position
            for w in range(-window_size, window_size + 1):
                context_word_pos = center_word_pos + w
            # make soure not jump out sentence
                if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                    continue
                context_word_idx = indices[context_word_pos]
                idx_pairs.append((indices[center_word_pos], context_word_idx))
                
    idx_pairs = np.array(idx_pairs) # it will be useful to have this as numpy array

def get_input_layer(word_idx):
    x = torch.zeros(pair_size).float()
    x[word_idx] = 1.0
    return x
