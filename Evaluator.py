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
import sys
from datashape.coretypes import float64
from decimal import getcontext
from expParams import expParameters
from mdpParams import mdpParameteres
import simplejson
import re

'''
Created on Jan 17, 2016

@author: mgomrokchi
'''
from simpleMC import MChain
from scipy.cluster.hierarchy import maxdists
class MCPE():
    def __init__(self, mdp, featureMatrix, policy, batch_gen_trigger="N", huge_batch_name="huge_batch.txt"):
        self.gamma=mdp.getGamma()
        self.MaxRewards=mdp.getMaxReward()
        self.pi=policy
        self.goalStates=mdp.getGoalstates()
        self.numStates=len(mdp.getStateSpace())
        self.featureMatrix=featureMatrix
        self.huge_batch_name=huge_batch_name
        self.batch_gen_trigger=batch_gen_trigger
        #To Do: 200 is set manually here, which is wrong, this needs to be fixed
        if batch_gen_trigger=="Y":
            self.InitHugeBatch= self.batchGen(mdp, 200, 5000, self.gamma, self.pi, mdp.startStateDistribution())
            self.batch_gen_trigger="N"
        #else:
        #    Batch_file = open(self.huge_batch_name, "w+")
        #   self.InitHugeBatch= Batch_file.readlines()
             
        
    
    def FirstVisit(self,trajectory, state, gamma):
        sIndexOfTau=0
        count=0
        reward=0
        temp=[]
        #finding the index of a state in the given trajectory 
        for i in trajectory:
            if state == i.split('-')[0]:
                sIndexOfTau=count
                break
            else:
                count=count+1   
        t=0          
        temp3=int(len(trajectory)-sIndexOfTau)
        while t < temp3 :
            if trajectory[t+sIndexOfTau] is '\n':
                break
            temp=trajectory[t+sIndexOfTau].split('-')
            reward = reward + int(temp[1])*pow(gamma,t)
            t=t+1
      
        return reward
                   
    def batchGen(self, MDP,maxTrajectoryLenghth ,numTrajectories, gamma=0.9, pi="uniform", inistStateDist="uniform"):

        if os.path.isfile(self.huge_batch_name) and self.batch_gen_trigger=="Y":
            input_var = input("The file "+ self.huge_batch_name+ " already exists, please enter a new file name for the new batch: ")
            print ("you entered " + input_var)
            Batch_file = open(input_var, "w+")
            self.huge_batch_name=input_var
        else:
            if self.batch_gen_trigger=="Y": 
                Batch_file = open(self.huge_batch_name, "w+")
        Batch = [[ ] for y in range(numTrajectories)]
        i=0
        while i < numTrajectories:  
            sourceState= MDP.sampleStartState()
            nextState=sourceState 
                            
            j=0 
            line=[]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
            while True:
                #for now it is not working with an input policy
                temp=MDP.getObsorbingStates()
                if int(sourceState)==int(temp):
                    r= MDP.getReward(sourceState, nextState)
                    nextState=int(nextState)
                    Batch[i].append([int(sourceState),r])
                    sourceState=int(sourceState)
                    Batch_file.write(str(sourceState)+'-'+str(r)+',')
                    j=0
                    break
                if j==maxTrajectoryLenghth:
                    j=0
                    break
                nextState=MDP.sampleNextState(sourceState)
                nextState=int(nextState)
                #here I have to generate the reward matrix associated to the MC and the get the reward w.r.t that but I am not doing in the current version
                r= MDP.getReward(sourceState, nextState)
                Batch[i].append([sourceState,r])
                sourceState=int(sourceState)
                Batch_file.write(str(sourceState)+'-'+str(r)+',')

                #Batch_file.write(';')
                sourceState=nextState   
                j=j+1  
            #line= numpy.asarray(line)
            Batch_file.write("\n")
            #simplejson.dump("\n", Batch_file)
            i=i+1  
        Batch_file.close()
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
                    j=j.split('-') 
                    if  j[0]!='\n' and s == int(j[0]):
                        tempFV = tempFV + self.FirstVisit(trajectory, s, myMDP.getGamma())
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
    
    def batchCutoff(self, filename, numTrajectories):
        miniBatch = [[ ] for y in range(numTrajectories)]
        batch_file = open(filename, "r")
        for i in range(numTrajectories):
            newLine=batch_file.readline()
            if newLine=="\n":
                newLine=batch_file.readline()
        #miniBatch= batch_file.readlines()
            #newlist= newLine.split (';')
            miniBatch[i]=newLine.split(',')
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

    def computeEligibilities(self, featureMatrix, lambda_coef, trajectory, gamma):
        dim=len(featureMatrix)
        upBound=len(trajectory)-1
        Z=numpy.zeros(shape=(upBound,dim))
        temp=numpy.zeros((dim,1))
        temp=numpy.reshape(temp,(dim,1))
        for i in range(upBound):
            temp=numpy.zeros(dim)
            temp=numpy.reshape(temp,(dim,1))
            for k in range(i):
                s_k=trajectory[k][0]
                temp2=featureMatrix[s_k,:]
                temp2=numpy.reshape(temp2, (dim,1))
                #temp2= temp2.T
                temp+=math.pow((lambda_coef*gamma),(i-k))*temp2
            for k in range(dim):
                Z[i,k]=temp[k]
                
        return Z
    
    def compute_LSTD_A_hat(self, trajectory, featureMatrix,gamma, lambdaCoef,stateSpace):
        dim= featureMatrix.shape[1]
        A_hat= numpy.zeros(shape=(len(stateSpace),dim))
        for i in range(len(trajectory)-1):
            s_i=trajectory[i][0]
            s_j=trajectory[i+1][0]
            phi_si=featureMatrix[s_i,:]
            phi_sj=gamma*featureMatrix[s_j,:]
            z_i=self.computeEligibilityVector(lambdaCoef, gamma, s_i, dim, featureMatrix)
            #z_i=z_i.T
            temp1 = phi_si-phi_sj
            temp1=temp1
            temp2=numpy.mat(z_i.T)*numpy.mat(temp1)
            A_hat+=temp2
        return 1/(len(trajectory))*A_hat
    
    def compute_LSTD_b_hat(self, trajectory, gamma, lambda_coef,featurematrix):
        dim=featurematrix.shape[1]
        b= numpy.zeros((1,dim))
        for i in range(len(trajectory)):
            b+=self.computeEligibilityVector(lambda_coef, gamma, i,dim,featurematrix)*trajectory[i][1]
        return b/(len(trajectory)-1)
    def computeEligibilityVector(self, lambdaCoef, gamma, index,dim,featureMatrix):
        z_i= numpy.zeros((1,dim))
        for j in range(index):
            phi_j=featureMatrix[j][:]
            temp=math.pow(gamma*lambdaCoef, index-j)
            z_i+=temp*(phi_j)
        return z_i
            
    
    def LSTD_lambda(self, featureMatrix, lambda_coef, mdp, trajectory):
        #eligabilityVMatrix=self.computeEligibilities(featureMatrix, lambda_coef, trajectory, mdp.getGamma())
        A_hat=self.compute_LSTD_A_hat(trajectory, featureMatrix, mdp.getGamma(), lambda_coef, mdp.getStateSpace())
        b_hat=self.compute_LSTD_b_hat(trajectory, mdp.getGamma(), lambda_coef, featureMatrix)
        A_hat=numpy.mat(A_hat)
        A_hat_inv=linalg.pinv(A_hat)
        theta_hat_X= numpy.mat(A_hat_inv)*numpy.mat(b_hat.T)
        return theta_hat_X
    
     
        
        
        