import numpy as np
import pandas as pd
from linear_classifier import LinearClassifier
from sentiment_reader import SentimentCorpus


class MultinomialNaiveBayes(LinearClassifier):

    def __init__(self):
        LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = True
        self.smooth_param = 1
        
    def train(self, x, y):
        # n_docs = no. of documents
        # n_words = no. of unique words    
        n_docs, n_words = x.shape

        self.x=x
        self.y=y
        z=range(n_docs)

        print(z)
        dataset=SentimentCorpus()
        arr=dataset.train_X
        #print(arr)
        print(n_docs)
        print(n_words)
        print(x.shape)
        
        #print(x[0:3])

        print(np.count_nonzero(y))
        R2=np.count_nonzero(y)
        print(np.count_nonzero(y==0))
        R1=np.count_nonzero(y==0)
        
        

        
        # classes = a list of possible classes
        self._classes = np.unique(y)
        
        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]
        
        # initialization of the prior and likelihood variables
        self._prior = np.zeros(n_classes, dtype=np.float64)
        likelihood = np.zeros((n_words,n_classes))

        #print(self.likelihood)

        ###########################
        #count = int(0)
        #temp_count = 0
        #x = open("positive.review","r")
        #print (x.readline())
        #arr = x.read().split()
        #print(arr[0])
        


        #for i in arr:
        #    if "positive" in i:
        #        temp_count+=1
        
        #print (temp_count)
        #if "positive" in x.read().split():
        #    count+=1
        #print (count)

        prior = np.zeros(n_classes, dtype=np.float64)
        
        prior=np.divide(prior+1,n_docs)

        freq_feature=np.zeros(n_classes)

        for doc,words in enumerate(x):
            for key,value in enumerate(words):
                likelihood[key,y[doc]]+=int(value)
                freq_feature[y[doc]]+=int(value)+dataset.nr_features

        likelihood[:,0] =np.divide(likelihood[:,0],freq_feature[0])
        likelihood[:,1] =np.divide(likelihood[:,1],freq_feature[1])

        ###########################

        params = np.zeros((n_words+1,n_classes))
        for i in range(n_classes): 
            # log probabilities
            params[0,i] = np.log(prior[i])
            with np.errstate(divide='ignore'): # ignore warnings
                params[1:,i] = np.nan_to_num(np.log(likelihood[:,i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params
