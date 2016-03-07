'''
Created on Mar 3, 2016

@author: mgomrokchi
'''

class expParameters(object):
    '''
    classdocs
    '''


    def __init__(self):
        
        self.maxTrajLength=200
        self.minNumberTraj=100
        self.numRounds=10
        self.experimentBatchLenghts=[10,50,100,500,1000,1500,2000,2500,3000,4000]#,5000] 
        self.epsilon=0.1
        self.delta=0.1
        self.regCoefs=[0.01,0.1,1,10,100,1000,10000]
        self.pow_exp=0.4
        self.aggregationFactor=1
        