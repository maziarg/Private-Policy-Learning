'''
Created on Mar 3, 2016

@author: mgomrokchi
'''

class MyClass(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        
        self.__maxTrajLength=200
        self.__minNumberTraj=100
        self.__numRounds=20
        self.__experimentBatchLenghts=[100,500,1000,1500,2000,2500,3000,3500,4000,4500,5000] 
        self.__epsilon=0.1
        self.__delta=0.05
        self.__regCoefs=[0.01,0.1,1,10,100,1000,10000]
        self.__pow_exp=0.4
        self.__aggregationFactor=1
        