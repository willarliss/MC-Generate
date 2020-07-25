import sys
import numpy as np
     

class MCGen:
    
    
    def __init__(self, order=1, reduce=False, minimum=None):
        
        assert order in [1, 2, 3], 'order of chain must be 1, 2, or 3'
        self.order = order
        
        if minimum is not None:
            assert minimum > 0, 'minimum length must be greater than 0' 
        self.minimum = minimum
        
        assert reduce in [True, False]
        self.reduce = reduce
    
    
    def build(self, corpus=[]):
        
        # Make sure the corpus is populated
        assert len(corpus) >= 1, 'corpus must be populated'
        
        # If minimum = None then user did not defnie minimum 
        if self.minimum == None:

            # Set minimum to minimum document length of corpus
            self.minimum = np.min(
                [len(doc.split()) for doc in corpus if doc.replace(' ', '') != '']
                )
                
        # Append the stopping character to the end of each document and remove empty documents
        corpus = [doc+' |||' for doc in corpus if doc.replace(' ', '') != '']        
        
        # Initialize the "transition matrix" and list of starting words
        self.chain_1, self.chain_2, self.chain_3 = {}, {}, {}
        self.start = []

        # Iterate over every document in the corpus
        for doc in corpus:
            doc = doc.split()

            # Add the first element of the list to the starting word list
            self.start.append(doc[0])
            
            # Always build first order markov transition matrix      
            pairs = [(doc[d], doc[d+1]) for d in np.arange(len(doc)-1)]
            for d1, d2 in pairs:
                if d1 in self.chain_1:
                    self.chain_1[d1].append(d2)
                else:
                    self.chain_1[d1] = [d2]
            
            if self.order == 2:
                # Build second order markov transition matrix if var order = 2        
                pairs = [(tuple(doc[d:d+2]), doc[d+2]) for d in np.arange(len(doc)-2)]
                for d1, d2 in pairs:
                    if d1 not in self.chain_2:
                        self.chain_2[d1] = [d2]
                    else:
                        self.chain_2[d1].append(d2)     
                        
            if self.order == 3:
                # Build third order markov transition matrix if var order = 3
                pairs = [(tuple(doc[d:d+3]), doc[d+3]) for d in np.arange(len(doc)-3)]
                for d1, d2 in pairs:
                    if d1 not in self.chain_3:
                        self.chain_3[d1] = [d2]
                    else:
                        self.chain_3[d1].append(d2) 

        if self.reduce:
            # Reduce transition matrix to remove probabilities of 1
            if self.order in [2, 3]:
                self.chain_2 = {k:v for k, v in self.chain_2.items() if len(set(v)) > 1}
                self.chain_3 = {k:v for k, v in self.chain_3.items() if len(set(v)) > 1}

        return self
                    
    
    def generate(self, n=1):
        
        # Make sure the number of iterations is greater than 0
        assert n >= 1, 'iterations must be at least 1'
            
        # Initialize the corpus for synthetic documents    
        synthetic_corpus = []
        
        # Make sure there is no seed 
        np.random.seed(None)
        
        # Iterate according to n parameter
        for r in np.arange(n):
            synthetic = [np.random.choice(self.start)]
            
            # Initialize indefinite loop
            while True:

                # Break the loop if the previous word is stopping character 
                if synthetic[-1] == '|||':
                    break
                
                if self.order == 1:
                    # Sample from first order chain
                    new = np.random.choice(self.chain_1[synthetic[-1]])
                    
                if self.order == 2:
                    try:
                        # Try sampling from the second order chain using the previous two words as index
                        new = np.random.choice(self.chain_2[tuple(synthetic[-2:])])
                    except KeyError:
                        # If second order chain doesnt work, use first order chain with previous word as index
                        new = np.random.choice(self.chain_1[synthetic[-1]])

                if self.order == 3:
                    try:
                        # Try sampling from the third order chain using the previous two words as index
                        new = np.random.choice(self.chain_3[tuple(synthetic[-3:])])
                    except KeyError:
                        # If third order chain doesnt work, use first order chain with previous word as index
                        new = np.random.choice(self.chain_1[synthetic[-1]])   
                        
                # If stopping character is produced but the document hasn't reached minimum length
                # This defaults to first order chain
                if  new == '|||' and len(synthetic) < self.minimum:  
                    
                    # If the index only points to stopping characters, start the document over
                    if set(self.chain_1[synthetic[-1]]) == {'|||'}:
                        synthetic = [np.random.choice(self.start)]
                        continue
                    
                    # Otherwise remove stopping characters and resample
                    else:
                        new = np.random.choice(
                            [i for i in self.chain_1[synthetic[-1]] if i != '|||']
                            )
                
                # Append most recent word to document
                synthetic.append(new)

            # Append document to synthetic corpus as string
            synthetic_corpus.append(' '.join(synthetic[:-1]))
            
        return np.array(synthetic_corpus)  
       
       
