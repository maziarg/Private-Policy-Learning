'''
Created on Feb 21, 2016

@author: mgomrokchi
'''
from Evaluator import MCPE
from simpleMC import MChain
import numpy
import matplotlib.pyplot as plt
from numpy import math, Inf, reshape
from scipy import  linalg
from decimal import Decimal
from sklearn.metrics import mean_squared_error
from Evaluator import MCPE

class experiment():
    def __init__(self, aggregationFactor,stateSpace,epsilon, delta, lambdaClass, numRounds, batchSize,policy):
        self.__Phi=self.featureProducer(aggregationFactor, stateSpace)
        self.__epsilon=epsilon
        self.__delta=delta
        self.__lambdaCalss=lambdaClass
        self.__numrounds=numRounds
        self.__batchSize=batchSize
        self.__policy=policy
        self.__stateSpace=stateSpace
        
    def lambdaExperiment_LSL(self, mdp,maxTrajectoryLenghth, regCoefs,pow_exp):
        myMCPE=MCPE(mdp,self.__Phi,self.__policy)
        V = myMCPE.realV(mdp)
        dim = len(self.__Phi)
        rho = mdp.startStateDistribution()
        maxR = mdp.getMaxReward()
        res=[]
        for k in range(self.__numrounds):
            S = myMCPE.batchGen(mdp, maxTrajectoryLenghth, self.__batchSize, mdp.getGamma(), self.__policy, rho)  
            FVMC = myMCPE.FVMCPE(mdp, self.__Phi, S)
            errls = []
            ridgeParams = myMCPE.computeLambdas(mdp, self.__Phi, regCoefs, self.__batchSize, pow_exp)
            for i in range(len(ridgeParams)):
                tL=myMCPE.LSL(FVMC[2], mdp, self.__Phi, ridgeParams[i], len(S))
                VL = self.__Phi*tL
                dpLSL=myMCPE.DPLSL(FVMC[2],FVMC[1], mdp, self.__Phi, mdp.getGamma(), self.__epsilon, self.__delta, ridgeParams[i], len(S), rho, self.__policy)
                temp5=reshape(dpLSL[0], (len(dpLSL[0]),1))
                dpVL = self.__Phi*temp5
                diff_V_VL=myMCPE.weighted_dif_L2_norm(mdp, V, VL)
                diff_V_dpVL=myMCPE.weighted_dif_L2_norm(mdp,V,dpVL)
                errls.append([ridgeParams[i], diff_V_VL, diff_V_dpVL])
                
            res.append([myMCPE.weighted_dif_L2_norm(mdp,V,FVMC[2]),errls])
        return res
    
    def TransitionFunction(self, sourceState, destState):
        if self.__policy is "uniform":
            if destState==sourceState and sourceState!=len(self.__stateSpace)-1:
                return 1.0/2
            if sourceState==destState-1 and sourceState!=len(self.__stateSpace)-1:
                return 1.0/2 
            if sourceState==len(self.__stateSpace)-1:
                if sourceState==destState:
                    return 1
                else:
                    return 0
            else: 
                return 0
        else:
            return 0
            
    def featureProducer(self, aggregationFactor, stateSpace):
        if aggregationFactor==1:
            return numpy.matrix(numpy.identity(len(stateSpace)), copy=False)
        else: 
            aggregatedDim=int(len(stateSpace)/aggregationFactor)
            aggFeatureMatrix=[[0 for col in range(len(stateSpace))] for row in range(int(aggregatedDim))]
            k=0
            for i in range(aggregatedDim):
                for j in range(len(stateSpace)):
                    if (j-i)-k==1 or j-i-k==0:
                        aggFeatureMatrix[i][j]=1
                    else:
                        aggFeatureMatrix[i][j]=0
                k=k+1       
            featureMatrix=numpy.reshape(aggFeatureMatrix,(aggregatedDim,len(stateSpace))) 
        return featureMatrix.T
    
    def rewardfunc (self, destState, goalstates, maxReward):
        if destState in goalstates:
            return maxReward
        else :
            return 0
    
        
def main():
    #######################MDP Parameters and Experiment setup###############################
    initDist="uniform"
    regCoefs=[0.01,0.1,1,10,100,1000,10000]
    pow_exp=0.4
    aggregationFactor=1
    numState= 20
    numAbsorbingStates=1
    #if the absorbing state is anything except 19 the trajectory will not terminate
    absorbingStates=[]
    absorbingStates.append(19)
    numGoalstates=1
    maxTrajLength=numState*10
    minNumberTraj=100
    numRounds=10
    lenTrajectories=[100,500,1000,1500,2000,2500,3000,3500,4000,4500,5000] 
    #discount factor
    gamma=0.9
    maxReward=1
    goalStates=[] 
    i=0
    while i < numGoalstates:
        sRand=numpy.random.randint(0,numState)
        if sRand not in goalStates:
            goalStates.append(sRand)
            i=i+1  
        else:
            continue     
    stateSpace=numpy.ones(numState)
    #To DO:Fix this part, since states should be in {0,1} 
    for i in range(numState):
        stateSpace[i]=i
    stateSpace=numpy.reshape(stateSpace, (numState,1))
    ##############################Privacy Parameters###############################
    epsilon=0.1
    delta=0.05
    ##############################MCPE Parameters##################################
    lambdaClass='L'
    policy="uniform"
    
    #####################Generating the feature matrix############################# 
    myExp=experiment(aggregationFactor,stateSpace,epsilon, delta, lambdaClass, numRounds, lenTrajectories[0],policy)
    featureMatrix=myExp.featureProducer(aggregationFactor, stateSpace)

    
    dim=len(featureMatrix)
    #Starting the MC-Chain construction
    myMDP = MChain(stateSpace, myExp.TransitionFunction, myExp.rewardfunc, goalStates, absorbingStates, gamma, maxReward)
  
    #Weight vector is used for averaging
    weightVector=[]
    for i in range(numState):
        if i==absorbingStates[0]:
            weightVector.append(0)
        else:
            weightVector.append(1/(numState-numAbsorbingStates))
    weightVector=numpy.reshape(weightVector,(numState,1))
    i=0
    result= myExp.lambdaExperiment_LSL(myMDP,maxTrajLength, regCoefs, pow_exp)

    
if __name__ == "__main__": main()