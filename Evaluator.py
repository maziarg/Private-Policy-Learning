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
    def __init__(self, mdp, featureMatrix, policy):
        self.gamma=mdp.getGamma()
        self.MaxRewards=mdp.getMaxReward()
        self.pi=policy
        self.goalStates=mdp.getGoalstates()
        self.numStates=len(mdp.getStateSpace())
        self.featureMatrix=featureMatrix
        
    
    def FirstVisit(self,trajectory, state, gamma):
        sIndexOfTau=0
        count=0
        reward=0
        temp=[]
        #finding the index of a state in the given trajectory 
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

    def FVMCPE(self,  myMDP, featuresMatrix, batch):
        # TODO: Make it incremental
        FV=[]
        Batch= batch
        S=myMDP.getStateSpace()
        for s in S: 
            #iterates through trajectories and search for state s 
            s=int(s[0])
            sBatchCount=0
            tempFV=0
            for i in range(len(batch)):
                trajectory=[]
                # Zero is used here due to the fact that Batch[i] is an array itself
                trajectory=Batch[i]  
                for j in trajectory:
                    #j[0] is the state and j[1] is the collected immediate reward                        
                    if  s == int(j[0]):
                        tempFV= tempFV + self.FirstVisit(trajectory, s, myMDP.getGamma())
                        sBatchCount=sBatchCount+1
                        break
                    else:
                        continue
            if sBatchCount==0:
                FV.append([s,0,0])
            else:
                tempFV=(tempFV/sBatchCount)           
                FV.append([s,tempFV,sBatchCount]) 
                sBatchCount=0
        firstVisitVecTemp=FV
        phiT=featuresMatrix.T        
        FirstVisitVector=[]
        stateApearanceCount=[]
        for i in firstVisitVecTemp:
            FirstVisitVector.append([i[1]])
            stateApearanceCount.append(i[2])
    
        invMatrix=numpy.mat(phiT)*numpy.mat(myMDP.getGammaMatrix())
        invMatrix=numpy.mat(invMatrix)*numpy.mat(featuresMatrix)
        invMatrix=invMatrix.I
        temp=numpy.mat(invMatrix)*numpy.mat(phiT)
        temp1=numpy.mat(temp)*numpy.mat(myMDP.getGammaMatrix())
        temp2=(numpy.reshape(FirstVisitVector, (len(FirstVisitVector),1)))
        ParamVec=numpy.mat(temp1)*numpy.mat(temp2)
        return [ParamVec, stateApearanceCount,FirstVisitVector]
    
    def SmootBound_LSW(self, myMDP, Gamma, countXVec, beta, startDist):
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
    def SmoothBound_LSL(self,featurmatrix, myMDP, countXVec, rho, regCoef,beta,numTrajectories):
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
               
    def DPLSW(self, thetaTild, countXVec, myMDP, featuresMatrix, gamma, epsilon, delta, initStateDist="uniform", pi="uniform"):
        alpha=15.0*numpy.sqrt(2*numpy.math.log(4.0/delta))
        alpha=alpha/epsilon 
        dim=len(featuresMatrix)
        beta= ((2*epsilon)/5)*math.pow((numpy.math.sqrt(dim)+math.sqrt(2*numpy.math.log(2.0/delta))),2)
        Gamma= myMDP.getGammaMatrix()
        GammaSqrt= linalg.sqrtm(Gamma)
        GammaSqrtPhi= numpy.mat(GammaSqrt) *numpy.mat(featuresMatrix)
        GammaSqrtPhiInv=GammaSqrtPhi.I 
        PsiBetaX= self.SmootBound_LSW(myMDP, Gamma, countXVec, beta, myMDP.startStateDistribution())
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
    
    def weighted_dif_L2_norm(self, mdp, v ,vhat):
        Gamma = mdp.getGammaMatrix()
        temp=numpy.mat((v-vhat).T)*numpy.mat(Gamma)*numpy.mat((v-vhat))
        temp=math.sqrt(temp)
        return temp


     
    def LSL(self,FirstVisitVector, myMDP,featuresMatrix,regCoef,numTrajectories):
        dim=len(featuresMatrix)
        phiT=featuresMatrix.T
        Gamma_X=myMDP.getGammaMatrix()
        invMatrix=numpy.mat(phiT)*numpy.mat(Gamma_X)
        invMatrix=numpy.mat(invMatrix)*numpy.mat(featuresMatrix)
        temp=regCoef/numTrajectories
        temp=(0.5*temp)
        Ident=temp*numpy.identity(dim)
        invMatrix= (invMatrix)+(Ident)
        invMatrix=invMatrix.I
        temp=numpy.mat(invMatrix)*numpy.mat(phiT)
        temp1=numpy.mat(temp)*numpy.mat(Gamma_X)
        temp2=[]
        for i in range(len(FirstVisitVector)):
            temp2.append(FirstVisitVector[i][0])
        temp2=(numpy.reshape(temp2, (len(temp2),1)))
        thetaTil_X=numpy.mat(temp1)*numpy.mat(temp2)
        return thetaTil_X
        
    def DPLSL (self, FirstVisitVector, countXVec, myMDP, featuresMatrix, gamma, epsilon, delta, regCoef, numTrajectories, rho, pi="uniform"):
        dim=len(featuresMatrix)
        Rho=numpy.reshape(rho,(len(rho),1))
        thetaTil_X= self.LSL(FirstVisitVector, myMDP, featuresMatrix, regCoef, numTrajectories)
        normPhi=numpy.linalg.norm(featuresMatrix)
        maxRho=numpy.linalg.norm(Rho,Inf)
        alpha=15.0*numpy.sqrt(2*numpy.math.log(4.0/delta))
        alpha=(alpha/epsilon) 
        beta= ((2*epsilon)/5)*math.pow((numpy.math.sqrt(dim)+math.sqrt(2*numpy.math.log(2.0/delta))),2)
        PsiBetaX= self.SmoothBound_LSL(featuresMatrix, myMDP, countXVec, myMDP.startStateDistribution(), regCoef, beta, numTrajectories)
        sigma_X=float(2*alpha*myMDP.getMaxReward()*normPhi/(1-myMDP.getGamma()))
        sigma_X=float(sigma_X/(regCoef-maxRho*numpy.math.pow(normPhi, 2)))
        sigma_X=sigma_X*PsiBetaX
        cov_X=math.pow(sigma_X,2)*numpy.identity(dim)
        mean=numpy.zeros(dim)
        ethaX=numpy.random.multivariate_normal(mean,cov_X)
        thetaTil_X=numpy.squeeze(numpy.asarray(thetaTil_X))
        ethaX=numpy.squeeze(numpy.asarray(ethaX))
        thetaTil_X_priv=thetaTil_X+ethaX
        return [thetaTil_X_priv, thetaTil_X,math.pow(sigma_X,2)]
    
    def newDPLSL (self, FirstVisitVector, countXVec, myMDP, featuresMatrix, gamma, epsilon, delta, regCoef, numTrajectories, rho, pi="uniform"):
        dim=len(featuresMatrix)
        Rho=numpy.reshape(rho,(len(rho),1))
        thetaTil_X= self.LSL(FirstVisitVector, myMDP, featuresMatrix, regCoef, numTrajectories)
        normPhi=numpy.linalg.norm(featuresMatrix)
        maxRho=numpy.linalg.norm(Rho,Inf)
        l2Rho=numpy.linalg.norm(Rho)
        #alpha=15.0*numpy.sqrt(2*numpy.math.log(4.0/delta))
        #alpha=(alpha/epsilon) 
        #beta= ((2*epsilon)/5)*math.pow((numpy.math.sqrt(dim)+math.sqrt(2*numpy.math.log(2.0/delta))),2)
        #PsiBetaX= self.SmoothBound_LSL(featuresMatrix, myMDP, countXVec, myMDP.startStateDistribution(), regCoef, beta, numTrajectories)
        sigma_X=float(2*myMDP.getMaxReward()*normPhi/(1-myMDP.getGamma()))
        sigma_X=float(sigma_X/(regCoef-maxRho*numpy.math.pow(normPhi, 2)))
        sigma_X=sigma_X*normPhi*maxRho*(math.sqrt(numTrajectories)+l2Rho)/(math.sqrt(2*regCoef))
        cov_X=math.pow(sigma_X,2)*numpy.identity(dim)
        mean=numpy.zeros(dim)
        ethaX=numpy.random.multivariate_normal(mean,cov_X)
        thetaTil_X=numpy.squeeze(numpy.asarray(thetaTil_X))
        ethaX=numpy.squeeze(numpy.asarray(ethaX))
        thetaTil_X_priv=thetaTil_X+ethaX
        return [thetaTil_X_priv, thetaTil_X,math.pow(sigma_X,2)]
    
    def realV(self,myMDP):
        R=myMDP.getExpextedRewardVec()
        P=myMDP.getTransitionMatix()            
        temp4=myMDP.getGamma()*P 
        temp5= numpy.identity(len(myMDP.getStateSpace()))
        temp4=temp5-temp4
        b = numpy.matrix(numpy.array(temp4))
        bInv=b.I 
        V_Real=numpy.mat(bInv)*numpy.mat(R)
        return V_Real
    
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
    
    def getminLambda(self, myMDP,featurmatrix):
        normPhi=numpy.linalg.norm(featurmatrix)
        l = normPhi**2
        l*=numpy.max(myMDP.startStateDistribution())
        return l
    def computeLambdas(self, myMDP, featurmatrix ,coefs,batchSize,p):
        lambdaS=[]
        lambdaOffset= self.getminLambda(myMDP, featurmatrix)
        for i in range(len(coefs)):
            lambdaS.append(lambdaOffset + coefs[i]*(batchSize**p))
        return lambdaS
