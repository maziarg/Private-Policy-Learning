'''
Created on Jun 6, 2016

@author: mgomrokchi
'''
import numpy as np
import math

class radialBasisFunctions(object):
    '''
        classdocs
    '''


    def __init__(self, stateSpace, meanList, varianceList):
        self.stateSpace=self.stateSpaceGen(stateSpace)
        self.meanList=meanList
        self.varianceList=varianceList
     
    def stateSpaceGen(self,stateSpace):
        return np.identity(len(stateSpace))
       
    def phiGen(self):
        Phi=[[0 for x in range(len(self.meanList))] for y in range(len(self.stateSpace))]
        for i in range(len(self.stateSpace)):
            Phi[i]=self.getBasis(self.stateSpace[i])
        Phi=np.reshape(Phi, (len(self.stateSpace),len(self.meanList)))  
        return Phi
    
    def getBasis(self,s):
        basisVec=np.zeros(len(self.meanList))
        for i in range(len(self.meanList)):
            basisVec[i]=self.gaussianBasis(s,self.meanList[i],self.varianceList[i])
        return basisVec
    def gaussianBasis(self, s, mean, variance):
        return math.exp((-np.linalg.norm(s-mean)**2)/(2*(variance**2)))
    
        
        
        
        