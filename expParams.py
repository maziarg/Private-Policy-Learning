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
        self.numRounds=5
        self.experimentBatchLenghts=[10]#, 500,1000,2000,4500,5000,10000,20000, 25000,30000,40000]#,5000] 
        self.epsilon=0.1
        self.delta=0.15
        self.delta_prime=0.1
        self.regCoefs = [0.1,1,10,100,1000,10000]
        self.pow_exp = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
        self.aggregationFactor=1
        self.lambdaCoef=[10000]
        self.numberOfSubsamples= 10
        self.distUB=10
        self.means=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
        self.sigmas=[0.001,0.01,0.1,1,10,100,1000,10000]