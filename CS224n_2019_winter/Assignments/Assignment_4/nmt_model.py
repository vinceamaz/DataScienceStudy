#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 4
nmt_model.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""
from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model_embeddings import ModelEmbeddings
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):
        """ Init NMT Model.

        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Hidden Size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(NMT, self).__init__()
        self.model_embeddings = ModelEmbeddings(embed_size, vocab)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        # default values
        self.encoder = None 
        self.decoder = None
        self.h_projection = None
        self.c_projection = None
        self.att_projection = None
        self.combined_output_projection = None
        self.target_vocab_projection = None
        self.dropout = None

 
        ### YOUR CODE HERE (~8 Lines)
        ### TODO - Initialize the following variables:
        ###     self.encoder (Bidirectional LSTM with bias)
        ###     self.decoder (LSTM Cell with bias)
        ###     self.h_projection (Linear Layer with no bias), called W_{h} in the PDF.
        ###     self.c_projection (Linear Layer with no bias), called W_{c} in the PDF.
        ###     self.att_projection (Linear Layer with no bias), called W_{attProj} in the PDF.
        ###     self.combined_output_projection (Linear Layer with no bias), called W_{u} in the PDF.
        ###     self.target_vocab_projection (Linear Layer with no bias), called W_{vocab} in the PDF.
        ###     self.dropout (Dropout Layer)
        ###
        ### Use the following docs to properly initialize these variables:
        ###     LSTM:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
        ###     LSTM Cell:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell
        ###     Linear Layer:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
        ###     Dropout Layer:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout
        
        ###################################################################################################################################################################
        ############
        ### Step 1: Initiate the entire Encoder network ###
        ############
        
        # nn.LSTM is a multi-layer long short-term memory (LSTM) RNN
            # input_size: The number of expected features in the input
            # hidden_size: The number of features in the hidden state
            # no need to specify number of time-steps
            # For the Encoder network, we use nn.LSTM because we only want the output of the entire network
        
        # Encoder network
            # Input of each bidirectional LSTM is the word embedding vector (shape 1 x e)
            # Output of each bidirectional LSTM is the hidden state and cell state of each LSTM (shape 1 x h)
            # The forward and backward LSTM outputs will be further concatenated resulting in shape (1 x 2h). The concatenation will be done seperately.
            
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, bias = True, bidirectional = True)
        
        ############
        ### Step 2: Initiate the structure of a LSTMCell for the Decoder network ###
        ############
        
        # nn.LSTMCell is a single long short-term memory (LSTM) cell (single time-step)
            # input_size: The number of expected features in the input
            # hidden_size: The number of features in the hidden state
            # For the Decoder network, we use nn.LSTMCell because need to compute the output prediction at each time-step (LSTM cell)
        
        # A LSTMCell for the Decoder network
            # The input will be the concatenation of the output vector from the previous LSTM time-step (shape 1 x h) and the input word embedding vector (shape 1 x e) at current step
            # The output will be the hidden state and cell state (both shape 1 x h in PDF)        
        
        self.decoder = nn.LSTMCell(input_size=hidden_size + embed_size, hidden_size=hidden_size, bias = True)
        
        ############
        ### Step 3: Initiate the Decoder network's first hidden state and cell state ###
        ############
        
        # We initiate the Decoder network's first hidden state and cell state with a linear projection (no activation) of the Encoder's final hidden state and final cell state
            # Linear projection means no activation, just multiplied by the weight matrix
        
        # Layer input: Concatenated bidirection hidden/cell state vector of the last layer of the Encoder network (1 x 2h)
        # W_{h/c}: Linear layer below (shape 2h x h)
        # Layer output: First hidden/cell state of Decoder network (shape 1 x h) 
        
        self.h_projection = nn.Linear(in_features=2*hidden_size, out_features=hidden_size, bias=False)
        self.c_projection = nn.Linear(in_features=2*hidden_size, out_features=hidden_size, bias=False)
        
        ############
        ### Step 4: Initiate the linear Attention Project Layer ###
        ############

        # We implement multiplicative attention (lecture slide# 78) through a linear layer
        
            # Layer input: the hidden state vector of Encoder network h_{enc} (shape src_len x 2h)
            # Layer output will be further multiplied by the hidden state of one Decoder time-step h_{dec}.T (shape 1 x h)
                # The multiplication result e is the attention score vector (size m x 1 in PDF)
            
            # Layer input h_{enc}: shape (src_len x 2h)
            # W_{att_projection}: Linear layer below (shape 2h x h)
            # Layer output: shape (src_len x h), which is a “liner projection” of the hidden state vector of the entire Encoding network
            
        self.att_projection = nn.Linear(in_features=2*hidden_size, out_features=hidden_size, bias=False)
        
        ############
        ### Step 5: Initiate the Attention Output Layer ###
        ############
        
        ## A softmax activation is not initialized here but will be done when the model object is built to normalize attention distribution for current Decoder LSTM step. The output of the softmax function is denoted as alpha_t in PDF (shape 1 x src_len) ##
        
        ## The output of the softmax function (shape 1 x src_len) will be multiplied by all the hidden states of the Encoder LSTM steps (shape src_len x 2h) and the result is the attention output for the current Decoder LSTM step (shape 1 x 2h)##  
        
        # We concatenate the attention output for the current Decoder LSTM step (shape 1 x 2h) and hidden state output of the current LSTM step in Decoder network (shape 1 x h) and run the result (shape 1 x 3h) through a linear layer to get the output vector of the current Decoder LSTM step
        
        # Layer input: concatenated attention output vector and hidden state of current Decoder time-step (shape 1 x 3h)
        # W_{u}: Linear layer below (shape 3h x h)
        # Layer output: V_t, almost the output of one Decoder time-step (shape 1 x h), still need to go through a couple of processes to be the final output of one Decoder time-step
        
        self.combined_output_projection = nn.Linear(3*hidden_size, hidden_size, bias=False)
        
        ############
        ### Step 6: Initiate Dropout ###
        ############ 
        
        self.dropout = nn.Dropout(p=self.dropout_rate)
        
        ############
        ### Step 7: Initiate a linear layer before the final softmax function ###
        ############
        
        # Then, we produce a probability distribution over target words at the current Decoder LSTM step through a softmax function
            # The softmax function is not initialized here but will be done when the model object is built
            # The output vector should go through a linear layer below before the softmax activation
        
        # Layer input: o_t, almost the output of one Decoder time-step (shape 1 x h)
        # W_{vocab}: Linear layer below (shape h x len(vocab.tgt))
        # Layer output: final prediction of one Decoder time-step (shape 1 x len(vocab.tgt)) -- one hot vector of the predicted word
        
        self.target_vocab_projection = nn.Linear(hidden_size, len(vocab.tgt), bias=False)
        
        ###################################################################################################################################################################
        

        ### END YOUR CODE


    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        @param source (List[List[str]]): list of source sentence tokens
        @param target (List[List[str]]): list of target sentence tokens, wrapped by `<s>` and `</s>`

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        # Compute sentence lengths
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)   # Tensor: (src_len, b)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)   # Tensor: (tgt_len, b)

        ###     Run the network forward:
        ###     1. Apply the encoder to `source_padded` by calling `self.encode()`
        ###     2. Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
        ###     3. Apply the decoder to compute combined-output by calling `self.decode()`
        ###     4. Compute log probability distribution over the target vocabulary using the
        ###        combined_outputs returned by the `self.decode()` function.
        
        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()
        
        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores


    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ Apply the encoder to source sentences to obtain encoder hidden states.
            Additionally, take the final states of the encoder and project them to obtain initial states for decoder.

        @param source_padded (Tensor): Tensor of padded source sentences with shape (src_len, b), where
                                        b = batch_size, src_len = maximum source sentence length. Note that 
                                       these have already been sorted in order of longest to shortest sentence.
        @param source_lengths (List[int]): List of actual lengths for each of the source sentences in the batch
        @returns enc_hiddens (Tensor): Tensor of hidden units with shape (b, src_len, h*2), where
                                        b = batch size, src_len = maximum source sentence length, h = hidden size.
        @returns dec_init_state (tuple(Tensor, Tensor)): Tuple of tensors representing the decoder's initial
                                                hidden state and cell.
        """
        enc_hiddens, dec_init_state = None, None

        ### YOUR CODE HERE (~ 8 Lines)
        ### TODO:
        ###     1. Construct Tensor `X` of source sentences with shape (src_len, b, e) using the source model embeddings.
        ###         src_len = maximum source sentence length, b = batch size, e = embedding size. Note
        ###         that there is no initial hidden state or cell for the decoder.
        ###     2. Compute `enc_hiddens`, `last_hidden`, `last_cell` by applying the encoder to `X`.
        ###         - Before you can apply the encoder, you need to apply the `pack_padded_sequence` function to X.
        ###         - After you apply the encoder, you need to apply the `pad_packed_sequence` function to enc_hiddens.
        ###         - Note that the shape of the tensor returned by the encoder is (src_len b, h*2) and we want to
        ###           return a tensor of shape (b, src_len, h*2) as `enc_hiddens`.
        ###     3. Compute `dec_init_state` = (init_decoder_hidden, init_decoder_cell):
        ###         - `init_decoder_hidden`:
        ###             `last_hidden` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
        ###             Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
        ###             Apply the h_projection layer to this in order to compute init_decoder_hidden.
        ###             This is h_0^{dec} in the PDF. Here b = batch size, h = hidden size
        ###         - `init_decoder_cell`:
        ###             `last_cell` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
        ###             Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
        ###             Apply the c_projection layer to this in order to compute init_decoder_cell.
        ###             This is c_0^{dec} in the PDF. Here b = batch size, h = hidden size
        ###
        ### See the following docs, as you may need to use some of the following functions in your implementation:
        ###     Pack the padded sequence X before passing to the encoder:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_padded_sequence
        ###     Pad the packed sequence, enc_hiddens, returned by the encoder:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pad_packed_sequence
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat
        ###     Tensor Permute:
        ###         https://pytorch.org/docs/stable/tensors.html#torch.Tensor.permute
        
        ###################################################################################################################################################################
        ############
        ### Step 1: Get word embeddings for the source sentences ###
        ############
        
        # nn.Embedding:
            # Input: torch.LongTensor (number of indices to extract per mini-batch, mini-batch size)
            # Output: (number of indices to extract per mini-batch, mini-batch size, embedding_dim)
            
        # source_padded: shape (max_seq_len, batch_size) = (src_len, b)
        
        # X: shape (max_seq_len, batch_size, embedding_dim) = (src_len, b, e)
            # Notice here the change between input and output of nn.Embedding is just the additional word embedding in the last dimension
            # X is essential the word embedding vectors of all Encoder time-steps for one mini-batch
            # X will act as the input to the Encoder network, self.encoder = nn.LSTM() defined in __init__()
            
        X = self.model_embeddings.source(source_padded)
        
        ############
        ### Step 2: Feed word embeddings into Encoding network to get output, last hidden state, and last cell state of Encoder network ###
        ############
        
        # pack_padded_sequence: https://github.com/HarshTrivedi/packing-unpacking-pytorch-minimal-tutorial
        
        # Remove padding from X so that when we later feed it into RNN, the paddings will not be computed as input to LSTM steps
            # packed_input.data.shape: (unpadded_sum_seq_len, embedding_dim)
            
        packed_input = pack_padded_sequence(X, torch.Tensor(source_lengths))
        
        # LSTM: https://stackoverflow.com/questions/48302810/whats-the-difference-between-hidden-and-output-in-pytorch-lstm
            # Inputs: input, (h_0, c_0)
                # If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
            # Outputs: output, (h_n, c_n)
                # h_n: shape (num_layers * num_directions, batch, hidden_dim) = (1 * 2, b, h)
                # c_n: shape (num_layers * num_directions, batch, hidden_dim) = (1 * 2, b, h)
                
        # last_hidden: shape (2, b, h)
        # last_cell: shape (2, b, h)
        # packed_output.data.shape : (unpadded_sum_seq_len, hidden_dim)
        
        packed_output, (last_hidden, last_cell) = self.encoder(packed_input)
        
        ############
        ### Step 3: Post dimensionality processing of output of Encoder network: enc_hiddens ###
        ############
        
        # Unpack output to gain padding
        # enc_hiddens.shape : (max_seq_len, batch_size, hidden_dim) = (src_len, b, 2h)
        enc_hiddens, _ = pad_packed_sequence(packed_output)
        
        # convert it to shape (batch_size, max_seq_len, hidden_dim) = (b, src_len, 2h)
        enc_hiddens = enc_hiddens.transpose(0, 1) 
        
        ############
        ### Step 4: Post dimensionality processing of last hidden state, and last cell state of Encoder network: last_hidden, last_cell ###
        ############
        
        # last_hidden and last_cell have shape (2, b, h). The 0-dim has number 2, meaning that it contains the hidden state or cell state of both directions. We need to manually concatenate them 
        
        # convert it to shape (batch_size, hidden_dim)
        # torch.cat(tensors, dim=0, out=None) → Tensor
            # dim: the dimension over which the tensors are concatenated
            # last_hidden: shape (2, b, h)
            # last_hidden[0], last_hidden[1]: shape (b, h)
            # output: shape (b, 2h)
        
        # Essentially, we are concatenating the two directional hidden states of the Encoder network and concatenating the two directional cell states of the Encoder network
        
        last_hidden = torch.cat((last_hidden[0], last_hidden[1]), 1)
        last_cell = torch.cat((last_cell[0], last_cell[1]), 1)
        
        ############
        ### Step 5: Compute the Decoder network's first hidden state and cell state with a linear projection of the Encoder's final hidden state and final cell state ###
        ############
        
        # last_hidden: shape (b, 2h)
        # init_decoder_hidden: shape (b, h)
        init_decoder_hidden = self.h_projection(last_hidden)
        init_decoder_cell = self.c_projection(last_cell)
        
        dec_init_state = (init_decoder_hidden, init_decoder_cell)
        
        ###################################################################################################################################################################
        
        
        ### END YOUR CODE

        return enc_hiddens, dec_init_state


    def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
                dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_padded: torch.Tensor) -> torch.Tensor:
        """Compute combined output vectors for a batch.

        @param enc_hiddens (Tensor): Hidden states (b, src_len, h*2), where
                                     b = batch size, src_len = maximum source sentence length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks (b, src_len), where
                                     b = batch size, src_len = maximum source sentence length.
        @param dec_init_state (tuple(Tensor, Tensor)): Initial state and cell for decoder
        @param target_padded (Tensor): Gold-standard padded target sentences (tgt_len, b), where
                                       tgt_len = maximum target sentence length, b = batch size. 

        @returns combined_outputs (Tensor): combined output tensor  (tgt_len, b,  h), where
                                        tgt_len = maximum target sentence length, b = batch_size,  h = hidden size
        """
        # Chop of the <END> token for max length sentences.
        target_padded = target_padded[:-1]

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []

        ### YOUR CODE HERE (~9 Lines)
        ### TODO:
        ###     1. Apply the attention projection layer to `enc_hiddens` to obtain `enc_hiddens_proj`,
        ###         which should be shape (b, src_len, h),
        ###         where b = batch size, src_len = maximum source length, h = hidden size.
        ###         This is applying W_{attProj} to h^enc, as described in the PDF.
        ###     2. Construct tensor `Y` of target sentences with shape (tgt_len, b, e) using the target model embeddings.
        ###         where tgt_len = maximum target sentence length, b = batch size, e = embedding size.
        ###     3. Use the torch.split function to iterate over the time dimension of Y.
        ###         Within the loop, this will give you Y_t of shape (1, b, e) where b = batch size, e = embedding size.
        ###             - Squeeze Y_t into a tensor of dimension (b, e). 
        ###             - Construct Ybar_t by concatenating Y_t with o_prev.
        ###             - Use the step function to compute the the Decoder's next (cell, state) values
        ###               as well as the new combined output o_t.
        ###             - Append o_t to combined_outputs
        ###             - Update o_prev to the new o_t.
        ###     4. Use torch.stack to convert combined_outputs from a list length tgt_len of
        ###         tensors shape (b, h), to a single tensor shape (tgt_len, b, h)
        ###         where tgt_len = maximum target sentence length, b = batch size, h = hidden size.
        ###
        ### Note:
        ###    - When using the squeeze() function make sure to specify the dimension you want to squeeze
        ###      over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        ###   
        ### Use the following docs to implement this functionality:
        ###     Zeros Tensor:
        ###         https://pytorch.org/docs/stable/torch.html#torch.zeros
        ###     Tensor Splitting (iteration):
        ###         https://pytorch.org/docs/stable/torch.html#torch.split
        ###     Tensor Dimension Squeezing:
        ###         https://pytorch.org/docs/stable/torch.html#torch.squeeze
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat
        ###     Tensor Stacking:
        ###         https://pytorch.org/docs/stable/torch.html#torch.stack
        ###################################################################################################################################################################
        ############
        ### Step 1: Feed Encoder hidden state into the attention projection layer to obtain the attention output ###
        ############
        
        # enc_hiddens_proj will be used for computing attention scores in later steps
        
        # enc_hiddens: shape (b, src_len, 2h)
        # W_{attProj}: shape (2h, h)
        # enc_hiddens_proj = enc_hiddens · W_{attProj}: shape (b, src_len, h)
        enc_hiddens_proj = self.att_projection(enc_hiddens) 
        
        ############
        ### Step 2: Get word embeddings for the target sentences for the Decoder network ###
        ############
        
        # Y: shape (tgt_len, b, e)        
        Y = self.model_embeddings.target(target_padded) 
        
        ############
        ### Step 3: Step through the time steps in the Decoder network ###
        ############
        
        # tensor.split() example: https://blog.csdn.net/weixin_44613063/article/details/89576810
        # tensor.split(size of each piece, dimension to be split)
        # here we split Y's 0-dim into pieces of size 1
        for Y_t in Y.split(1, dim=0):  
            ############
            ### Step 3.1: Remove redundant dimension from Y_t ###
            ############
            
            # torch.squeeze() example: https://jamesmccaffrey.wordpress.com/2019/07/02/the-pytorch-view-reshape-squeeze-and-flatten-functions/
                # In some sense a dimension with size 1 is useless. The squeeze() function eliminate any dimension that has size 1
                # We can also pass the argument to specify which dimension to squeeze
                # Y_t is the word embedding vectors of a mini-bath at one time-step
                # Y_t: shape (1, b, e) → Y_t_squeezed: shape (b, e)
            Y_t_squeezed = Y_t.squeeze(dim=0) 
            
            ############
            ### Step 3.2: Concatenate the word embedding input at current time-step and the predicted output at previous time-step to be the LSTM input of current time-step ###
            ############
            
            # torch.cat() example: https://blog.csdn.net/weixin_44613063/article/details/89576810
                # torch.cat(tensors to concatenate, dimension to concatenate) → Tensor
                # o_prev: the predicted output at previous Decoder time-step
                # o_prev was initialized as shape (b, h) with torch.zeros()
                # Y_t_squeezed: the word embedding input at current time-step, shape (b, e)
                # Ybar_t: concatenated LSTM input of current time-step, shape (b, e+h)
            Ybar_t = torch.cat((Y_t_squeezed, o_prev), dim=1)
            
            ############
            ### Step 3.3: Compute one forward step of the LSTM decoder, including the attention computation to get output and new hidden state at current time-step ###
            ############
            
            dec_state, o_t, _ = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
            
            ############
            ### Step 3.4: Update the Decoder output vector (that contains output of all Decoder time-steps) to incldue the output of the current time-step ###
            ############
            
            # combined_outputs is a list that stores Decoder outputs. when we finish each Decoder time-step, we append the output to it
            # o_t: output at current Decoder time-step, shape (b, h)
            combined_outputs.append(o_t)
            
            ############
            ### Step 3.5: Update variable that stores the output of previous time-step ###
            ############
            
            o_prev = o_t
            
        ############
        ### Step 4: Reshape the Decoder output vector combined_outputs to (tgt_len, b, h) ###
        ############
        
        # before stacking, the length of 0-dim of combined_outputs is the number of time-steps in Decoder network
        combined_outputs = torch.stack(combined_outputs, dim = 0)
        ###################################################################################################################################################################        
        
        ### END YOUR CODE

        return combined_outputs


    def step(self, Ybar_t: torch.Tensor,
            dec_state: Tuple[torch.Tensor, torch.Tensor],
            enc_hiddens: torch.Tensor,
            enc_hiddens_proj: torch.Tensor,
            enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the LSTM decoder, including the attention computation.

        @param Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
        @param dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
        @param enc_hiddens (Tensor): Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        @param enc_hiddens_proj (Tensor): Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                    where b = batch size, src_len = maximum source length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks shape (b, src_len),
                                    where b = batch size, src_len is maximum source length. 

        @returns dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's new hidden state, second tensor is decoder's new cell.
        @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
        @returns e_t (Tensor): Tensor of shape (b, src_len). It is attention scores distribution.
                                Note: You will not use this outside of this function.
                                      We are simply returning this value so that we can sanity check
                                      your implementation.
        """

        combined_output = None

        ### YOUR CODE HERE (~3 Lines)
        ### TODO:
        ###     1. Apply the decoder to `Ybar_t` and `dec_state`to obtain the new dec_state.
        ###     2. Split dec_state into its two parts (dec_hidden, dec_cell)
        ###     3. Compute the attention scores e_t, a Tensor shape (b, src_len). 
        ###        Note: b = batch_size, src_len = maximum source length, h = hidden size.
        ###
        ###       Hints:
        ###         - dec_hidden is shape (b, h) and corresponds to h^dec_t in the PDF (batched)
        ###         - enc_hiddens_proj is shape (b, src_len, h) and corresponds to W_{attProj} h^enc (batched).
        ###         - Use batched matrix multiplication (torch.bmm) to compute e_t.
        ###         - To get the tensors into the right shapes for bmm, you will need to do some squeezing and unsqueezing.
        ###         - When using the squeeze() function make sure to specify the dimension you want to squeeze
        ###             over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        ###
        ### Use the following docs to implement this functionality:
        ###     Batch Multiplication:
        ###        https://pytorch.org/docs/stable/torch.html#torch.bmm
        ###     Tensor Unsqueeze:
        ###         https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
        ###     Tensor Squeeze:
        ###         https://pytorch.org/docs/stable/torch.html#torch.squeeze
        ###################################################################################################################################################################
        ############
        ### Step 1: Apply the input (concatenation of word embedding input at current time-step and output at previous time-step) into Decoder LSTMCell to get new output at current time-step ###
        ############
        
        # LSTMCell
            # Inputs: input, (h_0, c_0)
            # Outputs: (h_1, c_1)
        
        # Ybar_t: concatenated LSTM input of current time-step, shape (b, e+h)
        # dec_state as input contains both hidden state and cell state, hidden state and cell state both are shape (b, h)
        # dec_state as output: shape (2, b, h)
        
        dec_state = self.decoder(Ybar_t, dec_state)
        
        ############
        ### Step 2: Split dec_state into its two parts (dec_hidden, dec_cell) ###
        ############
        
        # dec_hidden, dec_cell: shape (b, h)
        (dec_hidden, dec_cell) = dec_state
        
        ############
        ### Step 3: Compute attention score vector for the current time-step ###
        ############
        
        # We multiply the hidden state vector “projection” of the entire Encoding network by the hidden state of the current time-step in Decoder network to get the attention score vector for the current time-step
        
        # enc_hiddens_proj: shape (b, src_len, h)
        # dec_hidden: shape (b, h)
        # torch.unsqueeze(dec_hidden, 2): shape (b, h, 1)
        # torch.bmm(input, mat2, out=None) → Tensor
            # If input is a (b×n×m) tensor, mat2 is a (b×m×p) tensor, out will be a (b×n×p) tensor.
            # enc_hiddens_proj.bmm(dec_hidden.unsqueeze(2)): shape (b, src_len, 1)
        # e_t: shape (b, src_len)
        # e_t contains the attentions score of each time-step in Encoding network on the current one time-step in Decoder network
        e_t = enc_hiddens_proj.bmm(dec_hidden.unsqueeze(2)).squeeze(2)
        
        ###################################################################################################################################################################
    
        ### END YOUR CODE

        # Set e_t to -inf where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.byte(), -float('inf'))

        ### YOUR CODE HERE (~6 Lines)
        ### TODO:
        ###     1. Apply softmax to e_t to yield alpha_t
        ###     2. Use batched matrix multiplication between alpha_t and enc_hiddens to obtain the
        ###         attention output vector, a_t.
        #$$     Hints:
        ###           - alpha_t is shape (b, src_len)
        ###           - enc_hiddens is shape (b, src_len, 2h)
        ###           - a_t should be shape (b, 2h)
        ###           - You will need to do some squeezing and unsqueezing.
        ###     Note: b = batch size, src_len = maximum source length, h = hidden size.
        ###
        ###     3. Concatenate dec_hidden with a_t to compute tensor U_t
        ###     4. Apply the combined output projection layer to U_t to compute tensor V_t
        ###     5. Compute tensor O_t by first applying the Tanh function and then the dropout layer.
        ###
        ### Use the following docs to implement this functionality:
        ###     Softmax:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.functional.softmax
        ###     Batch Multiplication:
        ###        https://pytorch.org/docs/stable/torch.html#torch.bmm
        ###     Tensor View:
        ###         https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat
        ###     Tanh:
        ###         https://pytorch.org/docs/stable/torch.html#torch.tanh
        ###################################################################################################################################################################
        ############
        ### Step 1: Compute attention distribution alpha_t for the current time-step ###
        ############
        
        # Softmax converts all attentions scores into values between [0, 1] and add up to 1
        
        # e_t: shape (b, src_len)
        # alpha_t: shape (b, src_len)
        alpha_t = F.softmax(e_t, dim=1)
        
        ############
        ### Step 2: Compute attention output a_t for the current time-step ###
        ############
        
        # We multiply the attention distribution vector by the hidden state vector of the entire Encoding network to get the attention output for the current time-step in Decoder network
        
        # torch.bmm(input, mat2, out=None) → Tensor
            # If input is a (b×n×m) tensor, mat2 is a (b×m×p) tensor, out will be a (b×n×p) tensor.
            # alpha_t: shape (b, src_len)
            # alpha_t.unsqueeze(1): shape (b, 1, src_len)
            # enc_hiddens: shape (b, src_len, 2h)
            # alpha_t.unsqueeze(1).bmm(enc_hiddens): shape (b, 1, 2h)
        # a_t: shape (b, 2h)
        a_t = alpha_t.unsqueeze(1).bmm(enc_hiddens).squeeze(1)
        
        ############
        ### Step 3: Concatenate attention output a_t with the hidden state of current Decoder time-step ###
        ############
        
        # U_t contains information from both the hidden state of current Decoder time-step and the attention from the Encoder network
        
        # dec_hidden: shape (b, h)
        # U_t: shape (b, 3h)
        U_t = torch.cat((a_t, dec_hidden), dim=1)
        
        ############
        ### Step 4: We pass the concatenated result through a linear layer ###
        ############
        
        V_t = self.combined_output_projection(U_t)
        
        ############
        ### Step 5: We apply tanh activation for the linear layer output and apply dropout to obtain the combined output vector O_t ###
        ############
        
        O_t = self.dropout(torch.tanh(V_t))

        ###################################################################################################################################################################
        
        ### END YOUR CODE

        combined_output = O_t
        return dec_state, combined_output, e_t

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size. 
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.
        
        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)


    def beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.model_embeddings.target(y_tm1)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _  = self.step(x, h_tm1,
                                                      exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.model_embeddings.source.weight.device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.model_embeddings.embed_size, hidden_size=self.hidden_size, dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
