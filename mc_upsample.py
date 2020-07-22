import os
import sys
import numpy as np

class MCUpsampling:
    
    def __init__(self):
        return 
    
    def build(self, corpus=[]):
        
        assert len(corpus) > 1        
        corpus = [doc+' |||' for doc in corpus]

        self.chain, self.start = {}, []

        for doc in corpus:
                        
            doc = doc.split()
            
            self.start.append(doc[0])
            pairs = [(doc[d], doc[d+1]) for d in np.arange(len(doc)-1)]
            
            for d1, d2 in pairs:
                if d1 in self.chain:
                    self.chain[d1].append(d2)
                else:
                    self.chain[d1] = [d2]
                    
    def generate(self, n=1):
        
        assert n >= 1
        synthetic_text = []
        np.random.seed(None)
        
        for r in np.arange(n):
            synthetic = [np.random.choice(self.start)]
        
            while True:
                
                a = synthetic[-1]
                
                if a == '|||':
                    break
                else:
                    b = np.random.choice(self.chain[synthetic[-1]])
                    synthetic.append(b)
        
            synthetic_text.append(' '.join(synthetic[:-1]))
            
        return np.array(synthetic_text)
        