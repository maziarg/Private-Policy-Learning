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
from expParams import expParameters
from mdpParams import mdpParameteres

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
        
    def lambdaExperiment_LSL(self, mdp,batchSize,maxTrajectoryLenghth, regCoefs,pow_exp):
        myMCPE=MCPE(mdp,self.__Phi,self.__policy)
        V = myMCPE.realV(mdp)
        dim = len(self.__Phi)
        rho = mdp.startStateDistribution()
        maxR = mdp.getMaxReward()
        res=[]
        for k in range(self.__numrounds):
            S = myMCPE.batchGen(mdp, maxTrajectoryLenghth, batchSize, mdp.getGamma(), self.__policy, rho)  
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
    
    def newGS_LSL_experiments(self,batchSize,mdp,maxTrajectoryLenghth, regCoef,pow_exp):
        myMCPE=MCPE(mdp,self.__Phi,self.__policy)
        V = myMCPE.realV(mdp)
        rho = mdp.startStateDistribution()
        ridgeParam=myMCPE.computeLambdas(mdp, self.__Phi, regCoef, batchSize, pow_exp)
        err_new_lsl=[]
        for k in range(self.__numrounds):
            S = myMCPE.batchGen(mdp, maxTrajectoryLenghth, batchSize, mdp.getGamma(), self.__policy, rho)  
            FVMC = myMCPE.FVMCPE(mdp, self.__Phi, S)
            tL=myMCPE.LSL(FVMC[2], mdp, self.__Phi, ridgeParam[0], batchSize)
            VL = self.__Phi*tL
            dpLSL=myMCPE.newDPLSL(FVMC[2],FVMC[1], mdp, self.__Phi, mdp.getGamma(), self.__epsilon, self.__delta, ridgeParam[0], batchSize, rho, self.__policy)
            temp5=reshape(dpLSL[0], (len(dpLSL[0]),1))
            dpVL = self.__Phi*temp5
            diff_V_VL=myMCPE.weighted_dif_L2_norm(mdp, V, VL)
            diff_V_dpVL=myMCPE.weighted_dif_L2_norm(mdp,V,dpVL)
            err_new_lsl.append([ridgeParam[0],diff_V_VL, diff_V_dpVL])
        return err_new_lsl
        
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
    

def run_lambdaExperiment_LSL(experimentList,myMDP_Params,myExp_Params,myMDP):
        i=0
        expResults=[]
        for i in range(len(myExp_Params.experimentBatchLenghts)):
            expResults.append(experimentList[i].lambdaExperiment_LSL(myMDP,myExp_Params.experimentBatchLenghts[i],myExp_Params.maxTrajLength, myExp_Params.regCoefs, myExp_Params.pow_exp))
        ax = plt.gca()
        ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
        ax.set_xscale('log')
        realV_vs_FVMC=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
        i=0
        for i in range(len(myExp_Params.experimentBatchLenghts)):
            for j in range(myExp_Params.numRounds):
                realV_vs_FVMC[i]+=(expResults[i][j][0]/myExp_Params.numRounds)
        ax.plot(myExp_Params.experimentBatchLenghts,realV_vs_FVMC)
        plt.show()
def run_newGS_LSL_experiments(experimentList,myMDP_Params,myExp_Params,myMDP):   
    i=0
    regCoef=[0.5]
    expResults=[]
    for i in range(len(myExp_Params.experimentBatchLenghts)):
        expResults.append(experimentList[i].newGS_LSL_experiments(myExp_Params.experimentBatchLenghts[i],myMDP,myExp_Params.maxTrajLength, regCoef, myExp_Params.pow_exp))
    ax = plt.gca()
    ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
    ax.set_xscale('log')
    realV_vs_FVMC=[]
    LSL_vs_DPLSL=[]
    i=0
    for i in range(len(myExp_Params.experimentBatchLenghts)):
        temp1=[]
        temp2=[]
        for j in range(myExp_Params.numRounds):
            temp1.append(expResults[i][j][1])
            temp2.append(expResults[i][j][2])
        realV_vs_FVMC.append(temp1)
        LSL_vs_DPLSL.append(temp2)
    
    mean_realV_vs_FVMC=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    std_realV_vs_FVMC=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    blm=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    bldl=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    bldu=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    for j in range(len(myExp_Params.experimentBatchLenghts)):
        mean_realV_vs_FVMC[j] = numpy.average(realV_vs_FVMC[j])#blm
        std_realV_vs_FVMC [j] = numpy.std(realV_vs_FVMC[j])#bld
        bldu[j] = math.log10(mean_realV_vs_FVMC[j]+std_realV_vs_FVMC [j])-math.log10(mean_realV_vs_FVMC[j])
        bldl[j] = -math.log10(mean_realV_vs_FVMC[j]-std_realV_vs_FVMC [j])+math.log10(mean_realV_vs_FVMC[j])
        blm[j] = math.log10(mean_realV_vs_FVMC[j])
    
    mean_LSL_vs_FVMC=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    std_LSL_vs_FVMC=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    lsl_blm=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    lsl_bldl=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    lsl_bldu=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    for j in range(len(myExp_Params.experimentBatchLenghts)):
        mean_LSL_vs_FVMC[j] = numpy.average(LSL_vs_DPLSL[j])#lsl_blm
        std_LSL_vs_FVMC [j] = numpy.std(LSL_vs_DPLSL[j])#bld
        lsl_bldu[j] = math.log10(mean_LSL_vs_FVMC[j]+std_LSL_vs_FVMC [j])-math.log10(mean_LSL_vs_FVMC[j])
        lsl_bldl[j] = -math.log10(mean_LSL_vs_FVMC[j]-std_LSL_vs_FVMC [j])+math.log10(mean_LSL_vs_FVMC[j])
        lsl_blm[j] = math.log10(mean_LSL_vs_FVMC[j])
    
    ax.errorbar(myExp_Params.experimentBatchLenghts, blm,  bldu, bldl)
    ax.errorbar(myExp_Params.experimentBatchLenghts, lsl_blm,  lsl_bldu, lsl_bldl)
    plt.xscale('log')
    #plt.ylim((-10,10))
    plt.ylabel('ABS Difference')
    plt.xlabel('Batch-size')
    plt.legend(["LSL vs. DP","dpLSL vs. DP" ],loc=3)
    plt.title("epsilon= "+str(myExp_Params.epsilon)+", delta= "+str(myExp_Params.delta)+", \lambda= 0.5 m^"+str(myExp_Params.pow_exp))
    #ax.plot(myExp_Params.experimentBatchLenghts,realV_vs_FVMC)
    #ax.plot(myExp_Params.experimentBatchLenghts,LSL_vs_DPLSL)
    plt.show()

def main():
    #######################MDP Parameters and Experiment setup###############################
    myExp_Params=expParameters()
    myMDP_Params=mdpParameteres()
    #if the absorbing state is anything except 19 the trajectory will not terminate
    absorbingStates=[]
    absorbingStates.append(19)    
    goalStates=[] 
    i=0
    while i < myMDP_Params.numGoalstates:
        sRand=numpy.random.randint(0,myMDP_Params.numState)
        if sRand not in goalStates:
            goalStates.append(sRand)
            i=i+1  
        else:
            continue     
    stateSpace=numpy.ones(myMDP_Params.numState)
    #To DO:Fix this part, since states should be in {0,1} 
    for i in range(myMDP_Params.numState):
        stateSpace[i]=i
    stateSpace=numpy.reshape(stateSpace, (myMDP_Params.numState,1))
    ##############################Privacy Parameters###############################
    
    ##############################MCPE Parameters##################################
    lambdaClass='L'
    policy="uniform"
    
    #####################Generating the feature matrix#############################
    myExps=[] 
    for k in range(len(myExp_Params.experimentBatchLenghts)): 
        myExps.append(experiment(myExp_Params.aggregationFactor,stateSpace,myExp_Params.epsilon, myExp_Params.delta, lambdaClass, myExp_Params.numRounds, myExp_Params.experimentBatchLenghts[k],policy))
    featureMatrix=myExps[0].featureProducer(myExp_Params.aggregationFactor, stateSpace)
    
    
    dim=len(featureMatrix)
    #Starting the MC-Chain construction
    myMDP = MChain(stateSpace, myExps[0].TransitionFunction, myExps[0].rewardfunc, goalStates, absorbingStates, myMDP_Params.gamma, myMDP_Params.maxReward)
  
    #Weight vector is used for averaging
    weightVector=[]
    for i in range(myMDP_Params.numState):
        if i==absorbingStates[0]:
            weightVector.append(0)
        else:
            weightVector.append(1/(myMDP_Params.numState-myMDP_Params.numAbsorbingStates))
    weightVector=numpy.reshape(weightVector,(myMDP_Params.numState,1))
    #run_lambdaExperiment_LSL(myExps, myMDP_Params, myExp_Params, myMDP)
    run_newGS_LSL_experiments(myExps, myMDP_Params, myExp_Params, myMDP)
    
if __name__ == "__main__": main()