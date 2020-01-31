#!/usr/bin/env python

    #####################################################################################################
    # Note: the coding part of the assignment must be done after the written part
    # Refer to the written for the computation used in the coding
    #####################################################################################################


import numpy as np
import random

from utils.gradcheck import gradcheck_naive
from utils.utils import normalizeRows, softmax


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE
    s = 1.0/(1+np.exp(-x))

    ### END YOUR CODE

    return s


def naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors (rows of matrix) for all words in vocab
                      (U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    (dJ / dU)
    """

    ### YOUR CODE HERE

    ### Please use the provided softmax function (imported earlier in this file)
    ### This numerically stable implementation helps you avoid issues pertaining
    ### to integer overflow. 
    
    # Step 1: Compute y_hat
    
    # N: dimension of the vocab
    # d: dimension of the word vec
    
    # outsideVectors has all words in the vocab
    theta = np.dot(outsideVectors, centerWordVec) # shape: (N x 1)
    
    y_hat = softmax(theta) # shape: (N x 1)
    
    #####################################################################################################
    
    # Step 2: Compute crossentropy loss
    
    # label vector y[outsideWordIdx] = 1
    # we pick the outside words we care about using [outsideWordIdx] from the vocab
    loss = - np.log(y_hat[outsideWordIdx])
    
    #####################################################################################################
    
    # Step 3: Compute gradient of loss with respect to center word vector
    
    y_hat_minus_y = y_hat # shape: (N x 1)
    y_hat_minus_y[outsideWordIdx] -= 1 # shape: (N x 1)
    
    # y_hat_minus_y: (N x 1)
    # outsideVectors: (N x d)
    
    gradCenterVec = np.dot(y_hat_minus_y.T, outsideVectors) # shape: (1 x d)
    
    #####################################################################################################
    
    # Step 4: Compute gradient of loss with respect to outside word vectors

    # y_hat_minus_y: (N x 1)
    # centerWordVec: (d x 1)
    
    
    # Note that the shape of y_hat_minus_y is acutally (N, ) and the shape of centerWordVec is actually (d, )
    # [:, np.newaxis] converts y_hat_minus_y to shape (N, 1) and centerWordVec to shape (d, 1)
    
    gradOutsideVecs = np.dot(y_hat_minus_y[:, np.newaxis], centerWordVec[:, np.newaxis].T) # shape: (N x d)

    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """

    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices

    ### YOUR CODE HERE

    ### Please use your implementation of sigmoid in here.
    
    
    # Step 1: Get positive sample's outside word vector and K negative samples' outside word vectors
    
    ### outsideVectors: matrix of all outside word vectors
    ### outside_vec: positive sample's outside word vector
    ### neg_vec: K negative samples' outside word vectors

    # outsideVectors: (N x d)
    
    outside_vec = outsideVectors[outsideWordIdx] # shape: (1 x d)
    neg_vec = outsideVectors[negSampleWordIndices] # shape: (K x d)
    
    #####################################################################################################
    
    # Step 2: Compute important components z_outside and z_neg
    
    ### theta_outside: dot product of (positive sample's outside word vector, center word vector)
    ### theta_neg: dot product of (K negative samples' outside word vectors, center word vector)
    ### z_outside: sigmoid(theta_outside)
    ### z_neg: sigmoid(-theta_neg)
    
    # outside_vec: (1 x d)
    # centerWordVec: (1 x d)
    # neg_vec:(K x d)
    
    theta_outside = np.dot(outside_vec, centerWordVec.T) # shape: (1)
    theta_neg = np.dot(neg_vec, centerWordVec.T) # shape: (K x 1)
    
    z_outside  = sigmoid(theta_outside) # shape: (1)
    z_neg = sigmoid(- theta_neg) # shape: (K x 1)
    
    #####################################################################################################
    
    # Step 3: Compute negative sampling loss
    
    loss = - (np.log(z_outside) + np.sum(np.log(z_neg)))
    
    #####################################################################################################
    
    # Step 4: Compute gradient of loss with respect to center word vector
    
    # z_outside: (1)
    # outside_vec: (1 x d)
    # z_neg: (K x 1)
    # neg_vec: (K x d)
    
    gradCenterVec = - (1.0 - z_outside)*outside_vec + np.sum(np.dot((1 - z_neg).T, neg_vec)) # shape: (1 x d)
    
    #####################################################################################################
    
    # Step 5: Compute gradient of loss with respect to positive sample's outside word vector
    
    gradOutsideVecs = np.zeros_like(outsideVectors) # shape: (N x d)
    
    # z_outside: (1)
    # centerWordVec: (1 x d)
    
    gradOutsideVecs[outsideWordIdx] = - (1.0 - z_outside) * centerWordVec # shape:(1 x d)

    #####################################################################################################
    
    # Step 6: Compute gradient of loss with respect to K negative samples' outside word vectors    
    
    for i, neg_index in enumerate(negSampleWordIndices):
        
        # z_neg: (K x 1)
        # centerWordVec: (1 x d)
        
        gradOutsideVecs[neg_index] += (1 - z_neg[i]) * centerWordVec # remember negative can appear multiple times

    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) for all words in vocab
                        (V in pdf handout)
    outsideVectors -- outside word vectors (as rows) for all words in vocab
                    (U in pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVecs -- the gradient with respect to the center word vectors
            (dJ / dV in the pdf handout)
    gradOutsideVectors -- the gradient with respect to the outside word vectors
                        (dJ / dU in the pdf handout)
    """

    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)
    
    # gradCenterVecs: (N x d)
    # gradOutsideVectors: (N x d)

    ### YOUR CODE HERE
    
    # Step 1: Get current center word vector from input
    
    centerWordIdx = word2Ind[currentCenterWord] # int object
    centerWordVec = centerWordVectors[centerWordIdx] # shape: (d, )
    
    #####################################################################################################
    
    # Step 2: Get indices of outside word vectors corresponding to the current center word
    
    outsideWordIndices = [word2Ind[i] for i in outsideWords]
    
    #####################################################################################################
    
    # Step 3: Loop through all outside word vectors for the current center word to compute
    # (1) the total loss corresponding to the current center word
    # (2) the gradient with respect to the current center word vector
    # (3) the gradient with respect to the outside word vectors
    
    for outsideWordIdx in outsideWordIndices:
        one_loss, one_gradCenterVec, one_gradOutsideVecs = \
            word2vecLossAndGradient(centerWordVec, outsideWordIdx, outsideVectors, dataset) 
        loss += one_loss
        gradCenterVecs[centerWordIdx] += one_gradCenterVec
        gradOutsideVectors += one_gradOutsideVecs
    
    ### END YOUR CODE

    return loss, gradCenterVecs, gradOutsideVectors

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, word2Ind, wordVectors, dataset, 
                         windowSize,
                         word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N/2),:]
    outsideVectors = wordVectors[int(N/2):,:]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(
            centerWord, windowSize1, context, word2Ind, centerWordVectors,
            outsideVectors, dataset, word2vecLossAndGradient
        )
        loss += c / batchsize
        grad[:int(N/2), :] += gin / batchsize
        grad[int(N/2):, :] += gout / batchsize

    return loss, grad


def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    print("==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, naiveSoftmaxLossAndGradient),
        dummy_vectors, "naiveSoftmaxLossAndGradient Gradient")

    print("==== Gradient check for skip-gram with negSamplingLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingLossAndGradient),
        dummy_vectors, "negSamplingLossAndGradient Gradient")

    print("\n=== Results ===")
    print ("Skip-Gram with naiveSoftmaxLossAndGradient")

    print ("Your Result:")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
            *skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
                dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset) 
        )
    )

    print ("Expected Result: Value should approximate these:")
    print("""Loss: 11.16610900153398
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-1.26947339 -1.36873189  2.45158957]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
Gradient wrt Outside Vectors (dJ/dU):
 [[-0.41045956  0.18834851  1.43272264]
 [ 0.38202831 -0.17530219 -1.33348241]
 [ 0.07009355 -0.03216399 -0.24466386]
 [ 0.09472154 -0.04346509 -0.33062865]
 [-0.13638384  0.06258276  0.47605228]]
    """)

    print ("Skip-Gram with negSamplingLossAndGradient")   
    print ("Your Result:")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\n Gradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
        *skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:],
            dummy_vectors[5:,:], dataset, negSamplingLossAndGradient)
        )
    )
    print ("Expected Result: Value should approximate these:")
    print("""Loss: 16.15119285363322
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-4.54650789 -1.85942252  0.76397441]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
 Gradient wrt Outside Vectors (dJ/dU):
 [[-0.69148188  0.31730185  2.41364029]
 [-0.22716495  0.10423969  0.79292674]
 [-0.45528438  0.20891737  1.58918512]
 [-0.31602611  0.14501561  1.10309954]
 [-0.80620296  0.36994417  2.81407799]]
    """)

if __name__ == "__main__":
    test_word2vec()
