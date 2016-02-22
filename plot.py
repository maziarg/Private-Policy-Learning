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
def main():
    #######################Parameters###############################
    initDist="uniform"
    lambdaFactor=[0.01,0.1,1,10,100,1000,10000]
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
    #Privacy parameters
    epsilon=0.5
    delta=0.05
    featureAggregationFactor=1
    myMCPE = MCPE(numState)  
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
    for i in range(numState):
        stateSpace[i]=i
    stateSpace=numpy.reshape(stateSpace, (numState,1))
    #generating the feature matrix        
    featureMatrix= myMCPE.featureProducer(featureAggregationFactor, stateSpace)
    dim=len(featureMatrix)
    #Starting the MC Chain construction  and call
    myMDP = MChain(stateSpace, myMCPE.TransitionFunction, myMCPE.rewardfunc, goalStates, absorbingStates, gamma, maxReward)
    vMC_List = []
    vRealList =[]
    dif_Real_MC_List=[]
    dif_PLSW_MC_List=[]
    diffNonPriv2Real_List=[]
    dif_PLSW_Real_List=[]
    dif_PLSL_Real_List=[]
    sigma_PLSW_List=[]
    sigma_PLSL_List=[]
    #Weight vector is used for averaging
    weightVector=[]
    for i in range(numState):
        if i==absorbingStates[0]:
            weightVector.append(0)
        else:
            weightVector.append(1/(numState-numAbsorbingStates))
    weightVector=numpy.reshape(weightVector,(numState,1))
    V_MC= numpy.zeros(numState)
    i=0
    Batch=myMCPE.batchGen(myMDP,maxTrajLength,lenTrajectories[len(lenTrajectories)-1],gamma)
    
    while i < len(lenTrajectories):
        round_difRealMC=[]
        round_difPriv1Real=[]
        round_difPriv2Real=[]
        round_difRealNonPriv2=[]
        round_sigmaPriv1=[]
        round_sigmaPriv2=[]
        for k in range(0,numRounds):
            V_priv=0
            batch_i = myMCPE.batchCutoff(Batch, lenTrajectories[i])
            ValueCountVec=myMCPE.FirstVisitMCpolicyevaluation( myMDP, featureMatrix, batch_i, myMDP.getGamma(),lenTrajectories[i])
            theta_tild=ValueCountVec[0]
            countXVec=ValueCountVec[1]
            FirstVisitVector=ValueCountVec[2]
            theta_priveX=myMCPE.DPFVMC1(theta_tild,countXVec, myMDP, featureMatrix, gamma, epsilon, delta)
            theta_priveXv2=myMCPE.DPFVMC2(FirstVisitVector, countXVec, myMDP, featureMatrix, gamma, epsilon, delta, myMCPE.dynamicRegCoefGen(lambdaFactor[2],lenTrajectories[i],1), lenTrajectories[i], myMDP.startStateDistribution())
            round_sigmaPriv1.append(theta_priveX[2])
            round_sigmaPriv2.append(theta_priveXv2[2])
            ##########################computing Values#######################################
            V_MC=numpy.mat(featureMatrix)*numpy.mat(theta_tild)
            V_priv=numpy.mat(featureMatrix)*numpy.mat(numpy.reshape(theta_priveX[0], (dim,1)))
            V_priv2=numpy.mat(featureMatrix)*numpy.mat(numpy.reshape(theta_priveXv2[0], (dim,1)))
            V_priv2_nonprivate_part=numpy.mat(featureMatrix)*numpy.mat(numpy.reshape(theta_priveXv2[1], (dim,1)))
            
            #################computing real value estimates####################  
            R=myMDP.getExpextedRewardVec()
            P=myMDP.getTransitionMatix()            
            temp4=myMDP.getGamma()*P 
            temp5= numpy.identity(numState)
            temp4=temp5-temp4
            b = numpy.matrix(numpy.array(temp4))
            bInv=b.I 
            V_Real=numpy.mat(bInv)*numpy.mat(R)
            
            rmse_Real_nonPrive2_Vec=(numpy.math.sqrt(mean_squared_error(V_Real,V_priv2_nonprivate_part,weightVector)))
            rmse_Real_Prive2_Vec=(numpy.math.sqrt(mean_squared_error(reshape(V_Real, (numState,1)),reshape(V_priv2, (numState,1)),weightVector)))
            rmse_Real_Prive_Vec=(numpy.math.sqrt(mean_squared_error(V_Real,V_priv,weightVector)))
            rmse_Real_MC_Vec=(numpy.math.sqrt(mean_squared_error(V_Real,V_MC,weightVector)))
            
            
            round_difRealMC.append(rmse_Real_MC_Vec)
            round_difRealNonPriv2.append(rmse_Real_nonPrive2_Vec)
            round_difPriv1Real.append(rmse_Real_Prive_Vec)
            round_difPriv2Real.append(rmse_Real_Prive2_Vec)
            print("round",k,"  iteration", lenTrajectories[i])
            
          
        dif_Real_MC_List.append(numpy.average(round_difRealMC) )   
        diffNonPriv2Real_List.append(numpy.average(round_difRealNonPriv2))
        dif_PLSW_Real_List.append(numpy.average(round_difPriv1Real))
        dif_PLSL_Real_List.append(numpy.average(round_difPriv2Real))
        sigma_PLSW_List.append(numpy.average(round_sigmaPriv1))
        sigma_PLSL_List.append(numpy.average(round_sigmaPriv2))
        i=i+1
        
    l=0
    xList=[]
    while l < len(lenTrajectories):
        xList.append(l)
        l=l+1
    
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.plot(lenTrajectories,dif_Real_MC_List,'b')

    ax.legend(["FVMC vs. Dynamic Program","DPFVMC1. vs Dynamic Program.", "DPFVMC2 vs. Dynamic Program"],loc=1)
    plt.xlabel('Batch Size')
    plt.title("epsilon: "+ str(epsilon)+ "& delta: "+ str(delta)+ ' Reg. Coef.:'+ str(lambdaFactor[2])+ '* \sqrt(m)')
    plt.show()
    
    
#    print(sklearn.__check_build)
    
if __name__ == "__main__": main()