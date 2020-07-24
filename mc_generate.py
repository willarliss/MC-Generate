import os
import sys
import numpy as np

 
class MCGen:
    
    
    def __init__(self):
        
        pass
    
    
    def build(self, corpus=[]):
        
        # Make sure the corpus is populated
        assert len(corpus) > 1
        
        # Append the stopping character to the end of each document
        corpus = [doc+' |||' for doc in corpus]
        # Determine the minimum length of documets in the corpus
        self.minimum = np.min(
            [len(doc.split()) for doc in corpus]
            )
        
        # The minimum parameter must be greater than 0
        if self.minimum > 1:
            # If it can be afforded, subtract 1 from length to account for stopping character
            self.minimum -= 1
        
        # Initialize the "transition matrix" and list of starting words
        self.chain_1, self.chain_2, self.start = {}, {}, []

        # Iterate over every document in the corpus
        for doc in corpus:
            doc = doc.split()
    
            # If the first item of the document is not blank            
            if doc[0].strip != '':
                # Add the first element of the list to the starting word list
                self.start.append(doc[0])
            
            # Build first order markov transition matrix        
            pairs = [(doc[d], doc[d+1]) for d in np.arange(len(doc)-1)]
            for d1, d2 in pairs:
                if d1 in self.chain_1:
                    self.chain_1[d1].append(d2)
                else:
                    self.chain_1[d1] = [d2]
                    
            # Build second order markov transition matrix        
            pairs = [(tuple(doc[d:d+2]), doc[d+2]) for d in np.arange(len(doc)-2)]
            for d1, d2 in pairs:
                if d1 not in self.chain_2:
                    self.chain_2[d1] = [d2]
                else:
                    self.chain_2[d1].append(d2)                    
                    
        return self
                    
    
    def generate(self, n=1, minimum=None):
        
        # Make sure the number of iterations is greater than 0
        assert n > 0
    
        # Check if argument has been passed to minimum parameter
        if minimum is not None:
            # Make sure that argument is greater than 0
            assert minimum > 0
            # Override default minimum parameter defined in build method
            self.minimum = minimum
            
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
                
                try:
                    # Try sampling from the second order chain using the previous two words as index
                    new = np.random.choice(self.chain_2[tuple(synthetic[-2:])])
                except KeyError:
                    # If second order chain doesnt work, use first order chain with previous word as index
                    new = np.random.choice(self.chain_1[synthetic[-1]])
                    
                # If stopping character is produced but the document hasn't reached minimum length
                # This defaults to first order chain
                if  new == '|||' and len(synthetic) < self.minimum:  
                    
                    # If the index only points to stopping characters, start the document over
                    if set(self.chain_1[synthetic[-1]]) == {'|||'}:
                        synthetic = [np.random.choice(self.start)]
                        continue
                    
                    # Otherwise remove stopping characters and sample
                    else:
                        new = np.random.choice(
                            [i for i in self.chain_1[synthetic[-1]] if i != '|||']
                            )
                
                # Append most recent word to document
                synthetic.append(new)

            # Append document to synthetic corpus as string
            synthetic_corpus.append(' '.join(synthetic[:-1]))
            
        return np.array(synthetic_corpus)       

