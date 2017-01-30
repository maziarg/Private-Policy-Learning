from __future__ import division
import numpy
import numpy as np
import scipy
import matplotlib.pyplot as plt
from numpy import math, Inf, reshape, ravel
from scipy import  linalg
import os
from expParams import expParameters
from mdpParams import mdpParameteres
from scipy.spatial.distance import cdist, euclidean
from simpleMC import MChain

'''
Created on Jan 17, 2016

@author: mgomrokchi
'''


class MCPE():
    def __init__(self, mdp, featureMatrix, policy, batch_gen_trigger="N", huge_batch_name="huge_batch.txt"):
        self.gamma_factor=mdp.getGamma()
        self.MaxRewards=mdp.getMaxReward()
        self.pi=policy
        self.goalStates=mdp.getGoalstates()
        self.numStates=len(mdp.getStateSpace())
        self.featureMatrix=featureMatrix
        self.huge_batch_name=huge_batch_name
        self.batch_gen_trigger=batch_gen_trigger
        self.maxBatchLength=49999
        #To Do: 200 is set manually here, which is wrong, this needs to be fixed
        if batch_gen_trigger=="Y":
            self.InitHugeBatch= self.batchGen(mdp, 200, self.maxBatchLength, self.gamma_factor, self.pi, mdp.startStateDistribution())
            self.batch_gen_trigger="N"
            
    def FirstVisit(self,trajectory, state, gamma):
        sIndexOfTau=0
        count=0
        reward=0
        temp=[]
        for i in trajectory:
            if state == int(i.split('-')[0]):
                sIndexOfTau=count
                break
            else:
                count=count+1   
        t=0          
        while t < (len(trajectory)-sIndexOfTau) :
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
        print(i) 
        Batch_file.close()
        return Batch

    def FVMCPE(self,  myMDP, featuresMatrix, Batch):
        # TODO: Make it incremental
        FV=[]
        S=numpy.ravel(myMDP.getStateSpace())
        for s in S: 
            #iterates through trajectories and search for state s 
            sBatchCount=0
            tempFV=0
            for i in range(len(Batch)):
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
                tempFV/=sBatchCount           
                FV.append([s,tempFV,sBatchCount]) 
                sBatchCount=0
        firstVisitVecTemp=FV
        phiT=featuresMatrix.T        
        FirstVisitVector=[]
        stateApearanceCount=[]
        for i in firstVisitVecTemp:
            FirstVisitVector.append([i[1]])
            stateApearanceCount.append(i[2])
        FirstVisitVector=numpy.ravel(FirstVisitVector)
        Gamma_w=myMDP.getGammaMatrix()
        for i in range(len(stateApearanceCount)):
            Gamma_w[i][i]=Gamma_w[i][i]*stateApearanceCount[i]/len(Batch)
    
        invMatrix=(phiT*numpy.mat(Gamma_w))*featuresMatrix
        
        if self.is_invertible(invMatrix):
            invMatrix=linalg.inv(invMatrix)
            temp1=(numpy.mat(invMatrix)*(phiT))*Gamma_w
            temp2=(numpy.reshape(FirstVisitVector, (len(FirstVisitVector),1)))
            ParamVec=temp1*numpy.mat(temp2)
        else:
            for i in range(len(Gamma_w)):
                Gamma_w[i][i]=Gamma_w[i][i]**0.5
            invMatrix=numpy.mat(Gamma_w)*numpy.mat(featuresMatrix)
            invMatrix=linalg.pinv(invMatrix)
            ParamVec=numpy.mat((numpy.mat(invMatrix)*numpy.mat(Gamma_w)))*numpy.mat(numpy.reshape(FirstVisitVector, (len(FirstVisitVector),1)))
            
        
        return [ParamVec, stateApearanceCount,FirstVisitVector]
    
    def is_invertible(self,A):
        return A.shape[0] == A.shape[1] and numpy.linalg.matrix_rank(A) == A.shape[0]
    
    def varPhi_w(self,countXVec,k,Gamma):
        temp=0
        for s in range(len(countXVec)):
            templis=[1,countXVec[s]-k]
            temp+=Gamma[s][s]/((numpy.max(templis))**2)
        return temp
    
    def SmootBound_LSW(self, myMDP, Gamma, countXVec, beta, startDist):
        lInfty=int(numpy.linalg.norm(countXVec, Inf))

        Vals=[]
        for k in range(lInfty):
            Vals.append(self.varPhi_w(countXVec, k, Gamma)*math.exp(-k*beta))
        upperBound=numpy.max(Vals)  
        return upperBound
   
    def SmoothBound_LSL(self,featurmatrix, myMDP, countXVec, rho, regCoef,beta,numTrajectories):
        normPhi=numpy.linalg.norm(featurmatrix)
        maxRho=numpy.linalg.norm(rho,Inf)
        c_lambda=normPhi*maxRho/(math.sqrt(2*regCoef))
        #print('===============================================')
        #print(regCoef)
        l2Rho=numpy.linalg.norm(rho)
        phi_k=0
        Vals=[]
        for k in range(0,numTrajectories):
            minVal=0
            for s in range(len(myMDP.getStateSpace())-1):
                minVal=minVal+rho[s]*min(countXVec[s]+k,numTrajectories)
            phi_k=c_lambda*math.sqrt(minVal)+l2Rho
            Vals.append((math.pow(phi_k,2))*math.exp(-k*beta))    
        upperBound=max(Vals)
        #print('Smooth upper bound= '+str(upperBound))  
        return upperBound  
        
    def DPLSW(self, thetaTild, countXVec, myMDP, featuresMatrix, gamma, epsilon, delta,batchSize, initStateDist="uniform", pi="uniform"):
        dim=len(featuresMatrix.T)
        alpha=(5.0*numpy.sqrt(2*numpy.math.log(2.0/delta)))/epsilon
        beta= (epsilon/4)/(dim+numpy.math.log(2.0/delta))
        
        Gamma_w=myMDP.getGammaMatrix()
        for i in range(len(countXVec)):
            Gamma_w[i][i]=Gamma_w[i][i]*countXVec[i]/batchSize
        
        GammaSqrt= (Gamma_w)
        for i in range(len(GammaSqrt)):
            GammaSqrt[i][i]=math.sqrt(Gamma_w[i][i])
            
        GammaSqrtPhi= numpy.mat(GammaSqrt) *numpy.mat(featuresMatrix)
        if self.is_invertible(GammaSqrtPhi):
            GammaSqrtPhiInv=linalg.inv(GammaSqrtPhi)
        else:
            GammaSqrtPhiInv=linalg.pinv(GammaSqrtPhi)
            
        PsiBetaX= self.SmootBound_LSW(myMDP, Gamma_w, countXVec, beta, myMDP.startStateDistribution())
        sigmmaX= (alpha*myMDP.getMaxReward())/(1-self.gamma_factor)
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
        temp1=numpy.mat((numpy.ravel(v)-numpy.ravel(vhat)))*numpy.mat(Gamma)
        temp=temp1*numpy.mat(numpy.ravel(v)-numpy.ravel(vhat)).T
        temp=math.sqrt(temp)
        return temp

    def LSL(self,FirstVisitVector, myMDP,featuresMatrix,regCoef,numTrajectories,countVector):
        dim=len(featuresMatrix.T)
        phiT=featuresMatrix.T
        Gamma_X=myMDP.getGammaMatrix()
        I_count=numpy.identity(len(countVector))
        for i in range(len(countVector)):
            I_count[i,i]=countVector[i]
        Gamma_X=Gamma_X*I_count
        Gamma_X=Gamma_X/numTrajectories
        invMatrix=phiT*numpy.mat(Gamma_X)
        invMatrix=invMatrix*featuresMatrix
        temp=regCoef/numTrajectories
        temp=(0.5*temp)
        Ident=temp*numpy.identity(dim)
        invMatrix= (invMatrix)+numpy.mat(Ident)
        if self.is_invertible(invMatrix):
            invMatrix=linalg.inv(invMatrix)
        else:
            invMatrix=linalg.pinv(invMatrix)
        temp=numpy.mat(invMatrix)*numpy.mat(phiT)
        temp=temp*Gamma_X
        FirstVisitVector=numpy.reshape(FirstVisitVector, (len(FirstVisitVector),1))
        thetaTil_X=temp*FirstVisitVector     
        return thetaTil_X
        
    def DPLSL (self, LSL_Vector, countXVec, myMDP, featuresMatrix, gamma, epsilon, delta, regCoef, numTrajectories, rho, pi="uniform"):
        regCoef=regCoef
        dim=len(featuresMatrix.T)
        Rho=numpy.reshape(rho,(len(rho),1))
        thetaTil_X= LSL_Vector #self.LSL(FirstVisitVector, myMDP, featuresMatrix, regCoef, numTrajectories,countXVec)
        normPhi=numpy.linalg.norm(featuresMatrix)
        maxRho=numpy.linalg.norm(Rho,Inf)
        alpha=(5.0*numpy.sqrt(2*numpy.math.log(2.0/delta)))/epsilon
        beta= (epsilon/4)/(dim+numpy.math.log(2.0/delta))

        PsiBetaX = self.SmoothBound_LSL(featuresMatrix, myMDP, countXVec, myMDP.startStateDistribution(), regCoef, beta, numTrajectories)
        
        sigma_X=2*alpha*myMDP.getMaxReward()*normPhi/(1-myMDP.getGamma())
        sigma_X=sigma_X/(regCoef-maxRho*numpy.math.pow(normPhi, 2))
        sigma_X=sigma_X*(numpy.math.pow(PsiBetaX,0.5))
        
        #print(sigma_X)
        cov_X=math.pow(sigma_X,2)*numpy.identity(dim)
        mean=numpy.zeros(dim)
        ethaX=numpy.random.multivariate_normal(mean,cov_X)
        ethaX=numpy.reshape(ethaX,(len(ethaX),1))
        thetaTil_X_priv=thetaTil_X+ethaX
        return [thetaTil_X_priv, thetaTil_X,math.pow(sigma_X,2)]
    
    def GS_based_DPLSL (self, FirstVisitVector, countXVec, myMDP, featuresMatrix, gamma, epsilon, delta, regCoef, numTrajectories, rho, pi="uniform"):
        dim=len(featuresMatrix.T)
        Rho=numpy.reshape(rho,(len(rho),1))
        thetaTil_X= self.LSL(FirstVisitVector, myMDP, featuresMatrix, regCoef, numTrajectories,countXVec)
        
        
        normPhi=numpy.linalg.norm(featuresMatrix)
        maxRho=numpy.linalg.norm(Rho,Inf)
        l2Rho=numpy.linalg.norm(Rho)
        alpha=(5.0*numpy.sqrt(2*numpy.math.log(2.0/delta)))/epsilon

        sigma_X=float(2*alpha*myMDP.getMaxReward()*normPhi/((1-myMDP.getGamma())*(regCoef-maxRho*numpy.math.pow(normPhi, 2))))
        varphi_lamda=normPhi*maxRho*math.sqrt(numTrajectories)/(math.sqrt(2*regCoef))+l2Rho
        sigma_X=sigma_X*varphi_lamda
        cov_X=math.pow(sigma_X,2)*numpy.identity(dim)
        mean=numpy.zeros(dim)
        ethaX=numpy.random.multivariate_normal(mean,cov_X)
        #thetaTil_X=numpy.squeeze(numpy.asarray(thetaTil_X))
        #ethaX=numpy.squeeze(numpy.asarray(ethaX))
        thetaTil_X_priv=thetaTil_X+numpy.reshape(ethaX, (dim,1))
        return [thetaTil_X_priv, thetaTil_X,math.pow(sigma_X,2)]
    
    def realV(self,myMDP):
        R=myMDP.getExpextedRewardVec()
        P=myMDP.getTransitionMatix()
        gamma=myMDP.getGamma()           
        I= numpy.identity(len(myMDP.getStateSpace()))
        bInv=linalg.inv(I-gamma*numpy.mat(P)) 
        test_temp=bInv*(I-gamma*numpy.mat(P))
        V_Real=numpy.mat(bInv)*numpy.mat(R)
        return numpy.ravel(V_Real)
    
    def dynamicRegCoefGen(self, cFactor, numTrajectories,type):
        if type==1:
            temp=cFactor*(numpy.math.sqrt(numTrajectories))
            return temp
        else: 
            temp=cFactor*((numTrajectories))
            return temp
    
    def batchCutoff(self, filename, numTrajectories):
        miniBatch = [[ ] for y in range(numTrajectories)]
        randIndecies=numpy.random.choice(self.maxBatchLength, (1,numTrajectories), replace=False)
        batch_file = open(filename, "r")
        newbatch=self.picklines(batch_file, randIndecies[0])
        for i in range(numTrajectories):
            #newLine=batch_file.readline(randIndecies[i])
            newLine=newbatch[i]
            if newLine=="\n":
                lIndex=numpy.random.choice(self.maxBatchLength, replace=False)
                newLine=self.picklines(batch_file, lIndex) 
            miniBatch[i]=newLine.split(',')
        #Here I have to convert it to trajectories of ints and then return it
        return miniBatch 
    
    def picklines(self,thefile, whatlines):
        return [x for i, x in enumerate(thefile) if i in whatlines]
    
    def subSampleGen(self,batch, numberOfsubSamples, subSampelSize):

        subSamples=[]
        for i in range(numberOfsubSamples):
            subSamples.append(numpy.random.choice(batch, size=subSampelSize, replace=False))
        return subSamples
    def rDist(self,mdp,c, z,t_init, distance_upper_bound):#returns the t_init th distace of c to z and its value 
        distS=[]
        for i in range(len(z)):
            distS.append([linalg.norm(c-z[i]),i])
        a=sorted(distS, key=self.getKey)
        return [c,a[t_init-1][0]]
         
    def getKey(self, item):
        return item[0]
    def getKey2(self, item):
        return item[1][0]
    
    def geometric_median(self,X, eps=1e-5):
        y = np.mean(X, 0)

        while True:
            D = cdist(X, [y])
            nonzeros = (D != 0)[:, 0]
    
            Dinv = 1 / D[nonzeros]
            Dinvs = np.sum(Dinv)
            W = Dinv / Dinvs
            T = np.sum(W * X[nonzeros], 0)
    
            num_zeros = len(X) - np.sum(nonzeros)
            if num_zeros == 0:
                y1 = T
            elif num_zeros == len(X):
                return y
            else:
                R = (T - y) * Dinvs
                r = np.linalg.norm(R)
                rinv = 0 if r == 0 else num_zeros/r
                y1 = max(0, 1-rinv)*T + min(1, rinv)*y
    
            if euclidean(y, y1) < eps:
                return y1
    
            y = y1
            
               
    def aggregate_median(self,mdp,z,distUB):
        
        t_distS=[]
        t=int(len(z)/2)+1
        for i in range(len(z)):
            temp=self.rDist(mdp, z[i], z, t,distance_upper_bound=distUB)
            t_distS.append(temp)
        a=sorted(t_distS, key=self.getKey2)
        return [t_distS,a[0][0],a[1][0]]
    
    def generalized_median(self,mdp,z,t_int,distUB):
        rDistance=[]
        for i in range(len(z)):
            rDistance.append(self.rDist(mdp,z[i],z,t_int,distUB))
        mintemp=rDistance[0][1]
        #minIndex=rDistance[0][1][1]
        z_min=rDistance[0][0]
        for j in range(len(rDistance)):
            if mintemp>rDistance[j][1]:
                mintemp=rDistance[j][1]
                #minIndex=rDistance[j][1][1]
                z_min=rDistance[j][0]
        return [rDistance,z_min,mintemp]
    def computeRho(self,z,t,a,mdp,distance_upper_bound):
        rDistance=[]
        temp=0
        for i in range(len(z)):
            rDistance.append(self.rDist(mdp,z[i],z,t,distance_upper_bound))
        for i in range(a):
            if i >= len(z):
                break
            temp+=rDistance[i][1]
        if a==0:
            return rDistance[0][1]
        else: 
            return temp/a
    
    def computeAggregateSmoothBound(self, z,beta, s,mdp,distance_upper_bound):
        partitionPoint=int((len(z)+s)/2)
        a= int(s/beta)
        #rho=self.computeRho(partitionPoint, Dists,a)
        temp_1=0
        k=0
        t_0=partitionPoint+(k+1)*s
        tempList=[]
        #print(str(t_0)+" "+str(len(z)))
        while t_0 <=len(z):
            rho=self.computeRho(z, t_0 , a, mdp,distance_upper_bound)
            temp_2=rho*math.exp(-beta*k)
            tempList.append(temp_2)
            temp_1=max(temp_1, temp_2)
            k+=1
            t_0=partitionPoint+(k+1)*s
        #print(tempList)
        return 2*temp_1
    
    def LSW_subSampleAggregate(self, batch, numberOfsubSamples,myMDP,featuresMatrix,epsilon,delta,subSampleSize):
        dim=len(featuresMatrix.T)
        #alpha=(5.0*numpy.sqrt(2*numpy.math.log(2.0/delta)))/epsilon
        #beta= (epsilon/4)*(dim+numpy.math.log(2.0/delta))
        
        subSamples=self.subSampleGen(batch, numberOfsubSamples, subSampleSize)
        z=numpy.zeros((len(subSamples),len(featuresMatrix)))
        g=numpy.zeros((len(subSamples),len(featuresMatrix)))
        for i in range(len(subSamples)):
            FVMC=self.FVMCPE(myMDP, featuresMatrix, subSamples[i])
            DPLSWTemp= self.DPLSW(FVMC[0], FVMC[1], myMDP, featuresMatrix, myMDP.getGamma(), epsilon, delta, subSampleSize, "uniform", "uniform")
            g[i]= ravel(numpy.mat(featuresMatrix)*numpy.mat(FVMC[0]))
            z[i]= ravel(numpy.mat(featuresMatrix)*numpy.mat(DPLSWTemp[0]).T)#this is LSW
        tempAddz=numpy.zeros(dim)
        tempAddg=numpy.zeros(dim)
        
        for j in range(len(z)):
            tempAddz+=z[j]
            tempAddg+=g[j]
            
        
        #partitionPoint=int((numberOfsubSamples+math.sqrt(numberOfsubSamples))/2)+1   
        #g= self.generalized_median(myMDP,z,partitionPoint,distUB)
        #g=self.geometric_median(z)
        #g= self.aggregate_median(myMDP,z)
        
        #To check the following block
        #S_z=self.computeAggregateSmoothBound(z, beta, s,myMDP,distUB)
        #cov_X=(S_z/alpha)*numpy.identity(len(featuresMatrix))
        
        #ethaX=numpy.random.multivariate_normal(numpy.zeros(len(featuresMatrix)),cov_X)
        #print(S_z)
        #noise=(S_z/alpha)*ethaX
        #return [g[1]+ethaX,g[1]]
        return [tempAddz/len(subSamples),tempAddg/len(subSamples)]
    
    def LSL_subSampleAggregate(self, batch, s, numberOfsubSamples,myMDP,featuresMatrix,regCoef, pow_exp,numTrajectories,epsilon,delta,distUB):
        dim=len(featuresMatrix.T)
        alpha=(5.0*numpy.sqrt(2*numpy.math.log(2.0/delta)))/epsilon
        #beta= (epsilon/4)*(dim+numpy.math.log(2.0/delta))
        beta= (s/(2*numberOfsubSamples))
        
        subSamples=self.subSampleGen(batch, numberOfsubSamples)
        z=numpy.zeros((len(subSamples),dim))
        for i in range(len(subSamples)):
            FVMC=self.FVMCPE(myMDP, featuresMatrix, subSamples[i])
            #regc=self.computeLambdas(myMDP, featuresMatrix,regCoef, len(subSamples[i]), pow_exp)
            #z[i]=numpy.ravel(self.LSL(FVMC[2], myMDP, featuresMatrix,regc[0][0],len(subSamples[i]),FVMC[1]))
            SA_reidge_coef=100*math.pow(len(subSamples[i]), 0.5)
            z[i]=numpy.ravel(self.LSL(FVMC[2], myMDP, featuresMatrix,SA_reidge_coef,len(subSamples[i]),FVMC[1]))
            #z[i]= numpy.squeeze(numpy.asarray(FVMC[0]))#this is LSW
            
        partitionPoint=int((numberOfsubSamples+math.sqrt(numberOfsubSamples))/2)  
        g= self.generalized_median(myMDP,z,partitionPoint,distUB)
        #g= self.aggregate_median(myMDP,z)
        
        #To check the following block
        S_z=self.computeAggregateSmoothBound(z, beta, s,myMDP,distUB)
        #print(S_z)
        cov_X=(S_z/alpha)*numpy.identity((dim))
        ethaX=numpy.random.multivariate_normal(numpy.zeros(dim),cov_X)
        #print(S_z)
        #noise=(S_z/alpha)*ethaX
        #numpy.mat(featuresMatrix)*numpy.mat(g[1]+ethaX)
        temp_priv=numpy.mat(featuresMatrix)*numpy.mat(g[1]+ethaX).T
        return [temp_priv, numpy.mat(featuresMatrix)*numpy.mat(g[1]).T]
    
    
    def getminLambda(self, myMDP,featurmatrix):
        normPhi=numpy.linalg.norm(featurmatrix)
        l = normPhi**2
        l*=numpy.max(myMDP.startStateDistribution())
        return l
    def computeLambdas(self, myMDP, featurmatrix, coefs, batchSize , power_list):
        lambdaS=[]
        lambdaOffset= self.getminLambda(myMDP, featurmatrix)
        for p in power_list:
            for c in coefs:
                lambdaS.append([(lambdaOffset + c*(batchSize**p)),[c,p]])
        a=sorted(lambdaS, key=self.getKey)
        return a

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
    
     
        
        
        