from __future__ import division
import numpy 
import scipy
import matplotlib.pyplot as plt
from numpy import math, Inf, reshape
from scipy import  linalg
from decimal import Decimal
from sklearn.metrics import mean_squared_error
from matplotlib.cbook import todatetime
import os
import time
from datashape.coretypes import float64
from decimal import getcontext


'''
Created on Jan 17, 2016

@author: mgomrokchi
'''
from simpleMC import MChain
from scipy.cluster.hierarchy import maxdists
class MCPE():
    def __init__(self,numStates,goalStates=None,featureType='I', gamma=1,maxReward=1,policy="uniform"):
        self.gamma=gamma
        self.MaxRewards=maxReward
        self.pi=policy
        self.goalStates=goalStates
        self.numStates=numStates
        
    def TransitionFunction(self, sourceState, destState):
        if self.pi is "uniform":
            if destState==sourceState and sourceState!=self.numStates-1:
                return 1.0/2
            if sourceState==destState-1 and sourceState!=self.numStates-1:
                return 1.0/2 
            if sourceState==self.numStates-1:
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
    
    def FirstVisit(self,trajectory, state, gamma):
        sIndexOfTau=0
        count=0
        reward=0
        temp=[]
        for i in trajectory:
            if state == i[0]:
                sIndexOfTau=count
                break
            else:
                count=count+1   
        t=0          
        temp3=int(len(trajectory)-sIndexOfTau)
        while t < temp3 :
            temp=trajectory[t+sIndexOfTau]
            reward = reward + temp[1]*pow(gamma,t)
            t=t+1
      
        return reward
                   
    def batchGen(self, MDP,maxTrajectoryLenghth ,numTrajectories, gamma=0.9, pi="uniform", inistStateDist="uniform"):
        currentDirecoy=os.getcwd()
        currentFile=currentDirecoy+'/5000_trajectories'+str(time.strftime("%d"))
        f_Batch= open(currentFile, 'w')
        Batch = [[ ] for y in range(numTrajectories)]
        i=0
        while i < numTrajectories:  
            sourceState= MDP.sampleStartState()
            nextState=sourceState 
                            
            j=0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
            while True:
                #for now it is not working with an input policy
                temp=MDP.getObsorbingStates()
                if int(sourceState)==int(temp):
                    r= MDP.getReward(sourceState, nextState)
                    Batch[i].append([int(sourceState),r])
                    j=0
                    break
                if j==maxTrajectoryLenghth:
                    j=0
                    break
                nextState=MDP.sampleNextState(sourceState)
                #here I have to generate the reward matrix associated to the MC and the get the reward w.r.t that but I am not doing in the current version
                r= MDP.getReward(sourceState, nextState)
                Batch[i].append([int(sourceState),r])
                sourceState=nextState   
                j=j+1   
            f_Batch.write("%s\n" % Batch[i])          
            i=i+1  
        return Batch

    def FirstVisitMCpolicyevaluation(self,  myMDP, featuresMatrix, batch, gamma, numTrajectories, initStateDist="uniform", pi="uniform"):
        #RegList=[]
        #Values = []
        #ParamVec=[]
        FV=[]
        Batch= batch
        S=myMDP.getStateSpace()
        for s in S: 
            #iterates through trajectories and search for s 
            state=int(s)
            sBatchCount=0
            tempFV=0
            for i in range(numTrajectories):
                trajectory=[]
                trajectory=Batch[i][0]  
                for j in trajectory:
                    #j[0] is the state and j[1] is the collected immediate reward                        
                    if  state == j[0]:
                        tempFV= tempFV + self.FirstVisit(trajectory, s, gamma)
                        sBatchCount=sBatchCount+1
                        break
                    else:
                        continue
            if sBatchCount==0:
                FV.append([s,0,0])
            else:
                tempFV=(tempFV/sBatchCount)           
                FV.append([state,tempFV,sBatchCount]) 
        #Here I sort in order to make sure that when I multiply by feature matrix I respect the order of states       
        #firstVisitVecTemp=numpy.sort(FV,0)
        firstVisitVecTemp=FV
        phiT=featuresMatrix.T
        sSize=len(myMDP.getStateSpace())
        gammaMatrix= [[ ] for y in range(sSize)]
        #here I am using uniform initDist and therefore we use 1/myMDP.getState....
        FirstVisitVector=[]
        stateApearanceCount=[]
        for i in firstVisitVecTemp:
            FirstVisitVector.append([i[1]])
            stateApearanceCount.append(i[2])
        for i in range(len(myMDP.getStateSpace())):
            for j in range(len(myMDP.getStateSpace())):
                if i is j:
                    rho=float(1.0/sSize)
                    gammaMatrix[i].append(rho)
                else:
                    gammaMatrix[i].append(0)
                    
        invMatrix=numpy.mat(phiT)*numpy.mat(gammaMatrix)
        invMatrix=numpy.mat(invMatrix)*numpy.mat(featuresMatrix)
        invMatrix=invMatrix.I
        temp=numpy.mat(invMatrix)*numpy.mat(phiT)
        temp1=numpy.mat(temp)*numpy.mat(gammaMatrix)
        temp2=(numpy.reshape(FirstVisitVector, (len(FirstVisitVector),1)))
        ParamVec=numpy.mat(temp1)*numpy.mat(temp2)
        #ValueCountVec=[]
        #for i in range(len(myMDP.getStateSpace())):
        #    ValueCountVec.append([ParamVec[i],stateApearanceCount[i]])
        return [ParamVec, stateApearanceCount,FirstVisitVector]
    
    def ComputeSmoothBoundv1(self, myMDP, Gamma, countXVec, beta, startDist):
        lInfty=int(numpy.linalg.norm(countXVec, Inf))
        k=0
        Vals=[]
        for k in range(lInfty):
            temp1=math.exp(-k*beta)
            temp2=0
            for s in  range(len(myMDP.getStateSpace())-1):
                maxVal=max(countXVec[s]-k,1)
                maxValSq=math.pow(maxVal, 2)
                temp2=temp2+float(Gamma[s][s]/maxValSq)
            temp1=temp2*temp1
            Vals.append(temp1)
        upperBound=max(Vals)  
        return upperBound
    def ComputeSmoothBoundv2(self,featurmatrix, myMDP, countXVec, rho, regCoef,beta,numTrajectories):
        normPhi=numpy.linalg.norm(featurmatrix)
        maxRho=numpy.linalg.norm(rho,Inf)
        l2Rho=numpy.linalg.norm(rho)
        temp=0
        Vals=[]
        for k in range(0,numTrajectories):
            temp1=math.exp(-k*beta)
            temp2=0
            minVal=0
            for s in range(len(myMDP.getStateSpace())-1):
                temp2=min(countXVec[s]+k,numTrajectories)
                minVal=minVal+rho[s]*temp2
            temp=normPhi*maxRho*math.sqrt(minVal)
            temp=(temp/(math.sqrt(2)*regCoef))+l2Rho
            Vals.append(temp)
            
        upperBound=max(Vals)  
        return upperBound  
               
    def DPFVMC1(self, thetaTild, countXVec, myMDP, featuresMatrix, gamma, epsilon, delta, initStateDist="uniform", pi="uniform"):
        alpha=15.0*numpy.sqrt(2*numpy.math.log(4.0/delta))
        alpha=alpha/epsilon 
        dim=len(featuresMatrix)
        beta= ((2*epsilon)/5)*math.pow((numpy.math.sqrt(dim)+math.sqrt(2*numpy.math.log(2.0/delta))),2)
        Gamma= [[ ] for y in range(len(myMDP.getStateSpace()))]
        for i in range(len(myMDP.getStateSpace())):
            for j in range(len(myMDP.getStateSpace())):
                if i is j:
                    rho=(1.0/len(myMDP.getStateSpace()))
                    Gamma[i].append(rho)
                else:
                    Gamma[i].append(0)
            
        GammaSqrt= linalg.sqrtm(Gamma)
        GammaSqrtPhi= numpy.mat(GammaSqrt) *numpy.mat(featuresMatrix)
        GammaSqrtPhiInv=GammaSqrtPhi.I 
        PsiBetaX= self.ComputeSmoothBoundv1(myMDP, Gamma, countXVec, beta, myMDP.startStateDistribution())
        sigmmaX= (alpha*myMDP.getMaxReward())/(1-gamma)
        sigmmaX=sigmmaX*numpy.linalg.norm(GammaSqrtPhiInv)
        sigmmaX=sigmmaX*math.pow(PsiBetaX, .5)
        cov_X=math.pow(sigmmaX,2)*numpy.identity(dim)
        mean=numpy.zeros(dim)
        ethaX=numpy.random.multivariate_normal(mean,cov_X)        
        thetaTild=numpy.squeeze(numpy.asarray(thetaTild))
        ethaX=numpy.squeeze(numpy.asarray(ethaX))
        thetaTild_priv=thetaTild+ethaX
        return [thetaTild_priv,thetaTild,math.pow(sigmmaX,2)]
        
    def DPFVMC2 (self, FirstVisitVector, countXVec, myMDP, featuresMatrix, gamma, epsilon, delta, regCoef, numTrajectories, rho, pi="uniform"):
        currentDirecoy=os.getcwd()
        currentFile1=currentDirecoy+'/Theta_private'+' '+str(numTrajectories)+' '+str(epsilon)+str(delta)+' '+time.strftime("%d")
        f_tilda=open(currentFile1, 'w')
        f_tilda.write("__________________Reg Coef:"+str(regCoef)+"__________________________\n")

        dim=len(featuresMatrix)
        phiT=featuresMatrix.T
        Rho=numpy.reshape(rho,(len(myMDP.getStateSpace()),1))
        Gamma_X= [[ ] for y in range(len(myMDP.getStateSpace()))]
        for i in range(len(myMDP.getStateSpace())):
            for j in range(len(myMDP.getStateSpace())):
                if i == j:
                    temp=float(Rho[i])
                    temp=float(temp*countXVec[i])
                    rho=float(temp/numTrajectories)
                    Gamma_X[i].append(rho)
                else:
                    Gamma_X[i].append(0)
        invMatrix=numpy.mat(phiT)*numpy.mat(Gamma_X)
        invMatrix=numpy.mat(invMatrix)*numpy.mat(featuresMatrix)
        temp=regCoef/numTrajectories
        temp=(0.5*temp)
        Ident=temp*numpy.identity(dim)
        invMatrix= (invMatrix)+(Ident)
        invMatrix=invMatrix.I
        temp=numpy.mat(invMatrix)*numpy.mat(phiT)
        temp1=numpy.mat(temp)*numpy.mat(Gamma_X)
        temp2=(numpy.reshape(FirstVisitVector, (len(FirstVisitVector),1)))
        thetaTil_X=numpy.mat(temp1)*numpy.mat(temp2)
        normPhi=numpy.linalg.norm(featuresMatrix)
        maxRho=numpy.linalg.norm(Rho,Inf)
        alpha=15.0*numpy.sqrt(2*numpy.math.log(4.0/delta))
        alpha=(alpha/epsilon) 
        beta= ((2*epsilon)/5)*math.pow((numpy.math.sqrt(dim)+math.sqrt(2*numpy.math.log(2.0/delta))),2)
        PsiBetaX= self.ComputeSmoothBoundv2(featuresMatrix, myMDP, countXVec, myMDP.startStateDistribution(), regCoef, beta, numTrajectories)
        sigma_X=float(2*alpha*myMDP.getMaxReward()*normPhi/(1-myMDP.getGamma()))
        sigma_X=float(sigma_X/(regCoef-maxRho*numpy.math.pow(normPhi, 2)))
        sigma_X=sigma_X*PsiBetaX
        cov_X=math.pow(sigma_X,2)*numpy.identity(dim)
        mean=numpy.zeros(dim)
        ethaX=numpy.random.multivariate_normal(mean,cov_X)
        thetaTil_X=numpy.squeeze(numpy.asarray(thetaTil_X))
        ethaX=numpy.squeeze(numpy.asarray(ethaX))
        thetaTil_X_priv=thetaTil_X+ethaX
        for i in range(len(thetaTil_X_priv)):
            f_tilda.write("%s\n" % thetaTil_X_priv[i])
        
        return [thetaTil_X_priv, thetaTil_X,math.pow(sigma_X,2)]
    
    
    
    def dynamicRegCoefGen(self, cFactor, numTrajectories,type):
        if type==1:
            temp=cFactor*(numpy.math.sqrt(numTrajectories))
            return temp
        else: 
            temp=cFactor*((numTrajectories))
            return temp
    def batchCutoff(self, Batch, numTrajectories):
        miniBatch = [[ ] for y in range(numTrajectories)]
        for i in range(numTrajectories):
            newLine=Batch[i]
            miniBatch[i].append(newLine)
        return miniBatch 
