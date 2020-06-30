#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 4
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
"""

import torch.nn as nn

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layers.

        @param embed_size (int): Embedding size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size

        # default values
        self.source = None
        self.target = None

        src_pad_token_idx = vocab.src['<pad>']
        tgt_pad_token_idx = vocab.tgt['<pad>']

        ### YOUR CODE HERE (~2 Lines)
        ### TODO - Initialize the following variables:
        ###     self.source (Embedding Layer for source language)
        ###     self.target (Embedding Layer for target langauge)
        ###
        ### Note:
        ###     1. `vocab` object contains two vocabularies:
        ###            `vocab.src` for source
        ###            `vocab.tgt` for target
        ###     2. You can get the length of a specific vocabulary by running:
        ###             `len(vocab.<specific_vocabulary>)`
        ###     3. Remember to include the padding token for the specific vocabulary
        ###        when creating your Embedding.
        ###
        ### Use the following docs to properly initialize these variables:
        ###     Embedding Layer:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding
        
        # num_embeddings: how many unique words in the vocabulary
        # embedding_dim: dimension of the word embedding
        
        # This is a dictionary to look up word embedding of any word in the vocabuary by providing the index of the word
            # To visit the embedding of a word:
            # self.source(Variable(torch.LongTensor(idx)))
            # see https://yifdu.github.io/2018/12/05/Embedding%E5%B1%82/
            
        self.source = nn.Embedding(num_embeddings = len(vocab.src), embedding_dim = self.embed_size, padding_idx = src_pad_token_idx)
        self.target = nn.Embedding(num_embeddings = len(vocab.tgt), embedding_dim = self.embed_size, padding_idx = tgt_pad_token_idx)
        
        ### END YOUR CODE


