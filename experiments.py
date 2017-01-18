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
from IPython.core.tests.test_formatters import numpy
from radialBasis import  radialBasisFunctions
#from curses.has_key import python
#import seaborn as sns

class experiment():
    def __init__(self, aggregationFactor,stateSpace,epsilon, delta, lambdaClass, numRounds, batchSize,policy="uniform", batch_gen_param="N"):
        #self.__Phi=self.featureProducer(aggregationFactor, stateSpace)
        self.__Phi=self.featureProducer(aggregationFactor, stateSpace)
        self.__epsilon=epsilon
        self.__delta=delta
        self.__lambdaCalss=lambdaClass
        self.__numrounds=numRounds
        self.__batchSize=batchSize
        self.__policy=policy
        self.__stateSpace=stateSpace
        self.__batch_gen_param_trigger=batch_gen_param
    def getBatchSize(self):
        return self.__batchSize
    def radialfeature(self,stateSpace):
        myExpParams=expParameters()
        myradialBasis=radialBasisFunctions(stateSpace,myExpParams.means,myExpParams.sigmas)
        return myradialBasis.phiGen()
    
    def getPhi(self):
        return  self.__Phi  
    def getPolicy(self):
        return self.__policy
    
    
    def lambdaExperiment_SA_LSL(self, mdp,n_subsamples,batchsize,maxTrajectoryLenghth, regCoefs,pow_exp,s,epsilon,delta,Phi,distUB=10):
        myMCPE=MCPE(mdp,self.__Phi,self.__policy)
        V = myMCPE.realV(mdp)
        dim = len(numpy.mat(self.__Phi).T)
        rho = mdp.startStateDistribution()
        maxR = mdp.getMaxReward()
        res=[]
        for k in range(self.__numrounds):
            if (self.__batch_gen_param_trigger=="Y"):
                S = myMCPE.batchGen(mdp, maxTrajectoryLenghth, batchsize, mdp.getGamma(), self.__policy, rho)  
            else:
                S= myMCPE.batchCutoff("huge_batch.txt", batchsize)
            FVMC = myMCPE.FVMCPE(mdp, self.__Phi, S)
            errls = []
            ridgeParams_orig = myMCPE.computeLambdas(mdp, self.__Phi, regCoefs, int(batchsize/n_subsamples), pow_exp)
            ridgeParams=[]
            for l in range(len(ridgeParams_orig)):
                ridgeParams.append(ridgeParams_orig[l][0])
            for i in range(len(ridgeParams)):
                
                #tL=myMCPE.LSL(FVMC[2], mdp, self.__Phi, ridgeParams[i], len(S),FVMC[1])
                DPSA, SA =myMCPE.LSL_subSampleAggregate(S, s, n_subsamples, mdp, Phi, ridgeParams[i], pow_exp, batchsize, epsilon, delta, distUB) 
                DPSA=reshape(DPSA, (len(DPSA),1))
                SA=reshape(SA, (len(SA),1))
                diff_V_SA=myMCPE.weighted_dif_L2_norm(mdp, V, SA)
                diff_V_DPSA=myMCPE.weighted_dif_L2_norm(mdp,V,DPSA)
                errls.append([ridgeParams_orig[i][1], diff_V_SA, diff_V_DPSA])
            res.append(errls)
        return res
    
    def lambdaExperiment_LSL(self, mdp,batchSize,maxTrajectoryLenghth, regCoefs,pow_exp):
        myMCPE=MCPE(mdp,self.__Phi,self.__policy)
        V = myMCPE.realV(mdp)
        dim = len(numpy.mat(self.__Phi).T)
        rho = mdp.startStateDistribution()
        maxR = mdp.getMaxReward()
        res=[]
        for k in range(self.__numrounds):
            if (self.__batch_gen_param_trigger=="Y"):
                S = myMCPE.batchGen(mdp, maxTrajectoryLenghth, batchSize, mdp.getGamma(), self.__policy, rho)  
            else:
                S= myMCPE.batchCutoff("huge_batch.txt", batchSize)
            FVMC = myMCPE.FVMCPE(mdp, self.__Phi, S)
            errls = []
            ridgeParams_orig = myMCPE.computeLambdas(mdp, self.__Phi, regCoefs, self.__batchSize, pow_exp)
            ridgeParams=[]
            for l in range(len(ridgeParams_orig)):
                ridgeParams.append(ridgeParams_orig[l][0])
            for i in range(len(ridgeParams)):
                tL=myMCPE.LSL(FVMC[2], mdp, self.__Phi, ridgeParams[i], len(S),FVMC[1])
                VL = self.__Phi*tL
                dpLSL=myMCPE.DPLSL(tL,FVMC[1], mdp, self.__Phi, mdp.getGamma(), self.__epsilon, self.__delta, ridgeParams[i], len(S), rho, self.__policy)
                temp5=reshape(dpLSL[0], (len(dpLSL[0]),1))
                dpVL = self.__Phi*temp5
                diff_V_VL=myMCPE.weighted_dif_L2_norm(mdp, V, VL)
                diff_V_dpVL=myMCPE.weighted_dif_L2_norm(mdp,V,dpVL)
                errls.append([ridgeParams_orig[i][1], diff_V_VL, diff_V_dpVL])
                
            res.append(errls)
        return res
    
    def lambdaExperiment_GS_LSL(self, myMCPE, mdp,batchSize,maxTrajectoryLenghth, regCoefs,pow_exps):
        myMCPE=myMCPE
        V = myMCPE.realV(mdp)
        dim = len(self.__Phi)
        rho = mdp.startStateDistribution()
        maxR = mdp.getMaxReward()
        res=[]
        for k in range(self.__numrounds):
            if (self.__batch_gen_param_trigger=="Y"):
                S = myMCPE.batchGen(mdp, maxTrajectoryLenghth, batchSize, mdp.getGamma(), self.__policy, rho)  
            else:
                S= myMCPE.batchCutoff("huge_batch.txt", batchSize)
            FVMC = myMCPE.FVMCPE(mdp, self.__Phi, S)
            errls = []
            ridgeParams = myMCPE.computeLambdas(mdp, self.__Phi, regCoefs, self.__batchSize, pow_exps)
            for i in range(len(ridgeParams)):
                tL=myMCPE.LSL(FVMC[2], mdp, self.__Phi, ridgeParams[i], len(S))
                VL = self.__Phi*tL
                dpLSL_smoothed=myMCPE.DPLSL(FVMC[2],FVMC[1], mdp, self.__Phi, mdp.getGamma(), self.__epsilon, self.__delta, ridgeParams[i], len(S), rho, self.__policy)
                dpLSL_GS= myMCPE.GS_based_DPLSL(FVMC[2],FVMC[1], mdp, self.__Phi, mdp.getGamma(), self.__epsilon, self.__delta, ridgeParams[i], len(S), rho, self.__policy)
                temp5=reshape(dpLSL_smoothed[0], (len(dpLSL_smoothed[0]),1))
                temp6=reshape(dpLSL_GS[0], (len(dpLSL_GS[0]),1))
                dpVL_smoothed = self.__Phi*temp5
                dpVL_GS=self.__Phi*temp6
                diff_V_VL=myMCPE.weighted_dif_L2_norm(mdp, V, VL)
                diff_V_dpVL_smoothed=myMCPE.weighted_dif_L2_norm(mdp,V,dpVL_smoothed)
                diff_V_dpVL_GS=myMCPE.weighted_dif_L2_norm(mdp,V,dpVL_GS)
                errls.append([ridgeParams[i], diff_V_VL, diff_V_dpVL_smoothed,diff_V_dpVL_GS])
                
            res.append([myMCPE.weighted_dif_L2_norm(mdp,V,FVMC[2]),errls])
        return res
    
    #This is the experiment we run to compare
    def newGS_LSL_experiments(self,batchSize,mdp,maxTrajectoryLenghth, regCoef,pow_exp):
        myMCPE=MCPE(mdp,self.__Phi,self.__policy)
        V = myMCPE.realV(mdp)
        V=numpy.reshape(V, (len(V),1))
        rho = mdp.startStateDistribution()
        ridgeParam=myMCPE.computeLambdas(mdp, self.__Phi, regCoef, batchSize, pow_exp)
        err_new_lsl=[]
        for k in range(self.__numrounds):
            if (self.__batch_gen_param_trigger=="Y"):
                S = myMCPE.batchGen(mdp, maxTrajectoryLenghth, batchSize, mdp.getGamma(), self.__policy, rho)  
            else:
                S= myMCPE.batchCutoff("huge_batch.txt", batchSize)
            FVMC = myMCPE.FVMCPE(mdp, self.__Phi, S)
            tL=myMCPE.LSL(FVMC[2], mdp, self.__Phi, ridgeParam[0], batchSize,FVMC[1])
            VL = self.__Phi*tL
            dpLSL=myMCPE.GS_based_DPLSL(FVMC[2],FVMC[1], mdp, self.__Phi, mdp.getGamma(), self.__epsilon, self.__delta, ridgeParam[0], batchSize, rho, self.__policy)
            dpLSL_smoothed=myMCPE.DPLSL(FVMC[2],FVMC[1], mdp, self.__Phi, mdp.getGamma(), self.__epsilon, self.__delta, ridgeParam[0], batchSize, rho, self.__policy)
            #dpLSL=reshape(dpLSL[0], (len(dpLSL[0]),1))
            #dpLSL_smoothed=reshape(dpLSL_smoothed[0], (len(dpLSL_smoothed[0]),1))
            dpVL_GS = self.__Phi*dpLSL[0]
            dpVL_smoothed=self.__Phi*dpLSL_smoothed[0]
            diff_V_VL=myMCPE.weighted_dif_L2_norm(mdp, V, VL)
            diff_V_dpVLGS=myMCPE.weighted_dif_L2_norm(mdp,V,dpVL_GS)
            diff_V_dpVL_smoothed=myMCPE.weighted_dif_L2_norm(mdp,V, dpVL_smoothed)
            err_new_lsl.append([ridgeParam[0],diff_V_VL, diff_V_dpVLGS,diff_V_dpVL_smoothed])
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
    
    def LSW_subSampleAggregateExperiment(self, mdp,batchSize,maxTrajectoryLenghth,numberOfsubSamples,epsilon_star,delta_star, delta_prime,Phi,subSampleSize):   
        myMCPE=MCPE(mdp,self.__Phi,self.__policy)
        V = myMCPE.realV(mdp)
        rho = mdp.startStateDistribution()
        
        epsilon = math.log(0.5+math.sqrt(0.25+(batchSize*epsilon_star)/(subSampleSize*(math.sqrt(8*numberOfsubSamples*math.log(1/delta_prime))))))
        delta= (batchSize*(delta_star-delta_prime)/(subSampleSize*numberOfsubSamples))*(1/(0.5+math.sqrt(0.25+(batchSize*epsilon_star)/(subSampleSize*(math.sqrt(8*numberOfsubSamples*math.log(1/delta_prime)))))))
        
        resultsDPSA=[]
        resultsSA=[]
        temFVMC=[]
        DPLSW_result=[]
        tempMCPE=[0,0]
        for k in range(self.__numrounds):
            if (self.__batch_gen_param_trigger=="Y"):
                S = myMCPE.batchGen(mdp, maxTrajectoryLenghth, batchSize, mdp.getGamma(), self.__policy, rho)  
            else:
                S= myMCPE.batchCutoff("huge_batch.txt", batchSize)
            FVMC = myMCPE.FVMCPE(mdp, self.__Phi, S)
            DPLSW_result.append(numpy.mat(Phi)*numpy.mat(myMCPE.DPLSW(FVMC[0], FVMC[1], mdp, self.__Phi, mdp.getGamma(), epsilon_star, delta_star,batchSize)[0]).T)
            tempMCPE=myMCPE.LSW_subSampleAggregate(S, numberOfsubSamples,mdp,self.getPhi(),epsilon,delta,subSampleSize)
            resultsDPSA.append(tempMCPE[0])
            resultsSA.append(tempMCPE[1])
            temFVMC.append(numpy.mat(Phi)*numpy.mat(FVMC[0]))
            
        return [resultsDPSA,resultsSA,temFVMC,V,DPLSW_result]
    
    def LSL_subSampleAggregateExperiment(self, mdp, regCoef,batchSize,pow_exp,maxTrajectoryLenghth,numberOfsubSamples,s,epsilon,delta,Phi,distUB):   
        myMCPE=MCPE(mdp,self.__Phi,self.__policy)
        V = myMCPE.realV(mdp)
        rho = mdp.startStateDistribution()
        #ridgeParam=myMCPE.computeLambdas(mdp, self.__Phi, [regCoef], batchSize, pow_exp[0])
        resultsDPSA=[]
        resultsSA=[]
        temLSL=[]
        tempDPLSL=[]
        for k in range(self.__numrounds):
            if (self.__batch_gen_param_trigger=="Y"):
                S = myMCPE.batchGen(mdp, maxTrajectoryLenghth, batchSize, mdp.getGamma(), self.__policy, rho)  
            else:
                S= myMCPE.batchCutoff("huge_batch.txt", batchSize)
            FVMC = myMCPE.FVMCPE(mdp, self.__Phi, S)
            #ridgeParam=myMCPE.computeLambdas(mdp, self.__Phi, regCoef, len(S), pow_exp)
            lsl_reidge=10000*math.pow(len(S), 0.4)
            LSL_result=myMCPE.LSL(FVMC[2], mdp, self.__Phi, lsl_reidge, len(S),FVMC[1])
            DPLSL_result=myMCPE.DPLSL(LSL_result, FVMC[1], mdp, self.__Phi, mdp.getGamma(), epsilon, delta, lsl_reidge, len(S), rho)[0]
            #print('LSL Norm: '+str(numpy.linalg.norm(LSL_result)))
            #print('DPLSL Norm: '+str(numpy.linalg.norm(DPLSL_result)))
            tempSA=myMCPE.LSL_subSampleAggregate(S, s, numberOfsubSamples,mdp,self.getPhi(), regCoef, pow_exp, batchSize, epsilon,delta,distUB)
            resultsDPSA.append(tempSA[0])
            resultsSA.append(tempSA[1])
            temLSL.append(numpy.mat(Phi)*numpy.mat(LSL_result))
            tempDPLSL.append(numpy.mat(Phi)*numpy.mat(DPLSL_result))
        return [resultsDPSA,resultsSA,temLSL,V,tempDPLSL]
               
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
    #ax.set_xscale('log')
    ax.set_yscale('log')
    num_reidge_params=len(myExp_Params.regCoefs)*len(myExp_Params.pow_exp)

    #Real_vs_LSL_list=numpy.zeros(len(num_reidge_params))
    #Real_vs_DPLSL_list=numpy.zeros(len(num_reidge_params))
    expReal_vs_LS=numpy.zeros((len(myExp_Params.experimentBatchLenghts),num_reidge_params))
    expReal_vs_DPLSL=numpy.zeros((len(myExp_Params.experimentBatchLenghts),num_reidge_params))
    reidgeParamLsit=[]
    i=0
    num_reidge_params=len(myExp_Params.regCoefs)*len(myExp_Params.pow_exp)
    for i in range(len(myExp_Params.experimentBatchLenghts)):
        Real_vs_LSL_list=numpy.zeros((num_reidge_params))
        Real_vs_DPLSL_list=numpy.zeros((num_reidge_params))
        for j in range(myExp_Params.numRounds):
            tempLSL=[]
            tempDPLSL=[]
            reidgeParamLsit=[]
            for k in range(num_reidge_params):
                reidgeParamLsit.append(expResults[i][j][k][0])
                tempLSL.append(expResults[i][j][k][1])
                tempDPLSL.append(expResults[i][j][k][2])
            Real_vs_LSL_list+=numpy.ravel((1/myExp_Params.numRounds)*numpy.mat(tempLSL))
            Real_vs_DPLSL_list+=numpy.ravel((1/myExp_Params.numRounds)*numpy.mat(tempDPLSL))
        expReal_vs_LS[i]=Real_vs_LSL_list
        expReal_vs_DPLSL[i]=Real_vs_DPLSL_list
        
        
        
    #ax.plot(numpy.ravel(expReal_vs_LS[0]))
    #ax.plot(numpy.ravel(expReal_vs_DPLSL[0]))
    ax.plot(numpy.ravel(expReal_vs_LS[0]))
    ax.plot(numpy.ravel(expReal_vs_DPLSL[0]))
    #print(reidgeParamLsit[numpy.argmin(numpy.ravel(expReal_vs_LS[1])-numpy.ravel(expReal_vs_DPLSL[1]))])
    ax.plot(numpy.ravel(expReal_vs_LS[1]))
    ax.plot(numpy.ravel(expReal_vs_DPLSL[1]))
    #print(reidgeParamLsit[numpy.argmin(numpy.ravel(expReal_vs_LS[2])-numpy.ravel(expReal_vs_DPLSL[3]))])
    ax.plot(numpy.ravel(expReal_vs_LS[2]))
    ax.plot(numpy.ravel(expReal_vs_DPLSL[2]))
    #print(reidgeParamLsit[numpy.argmin(numpy.ravel(expReal_vs_LS[3])-numpy.ravel(expReal_vs_DPLSL[3]))])
    print(reidgeParamLsit[28])
    plt.show()

def run_lambdaExperiment_GS_LSL(myMCPE, experimentList,myMDP_Params, myExp_Params, myMDP):
    i=0
    
    meta_exponenet_test_Reuslts=[]
    for m in range(len(myExp_Params.pow_exp)):
        expResults=[]
        for i in range(len(myExp_Params.experimentBatchLenghts)):
            expResults.append(experimentList[i].lambdaExperiment_GS_LSL(myMCPE, myMDP,myExp_Params.experimentBatchLenghts[i],myExp_Params.maxTrajLength, myExp_Params.regCoefs, myExp_Params.pow_exp[m]))
        meta_exponenet_test_Reuslts.append(expResults)
    ax = plt.gca()
    #ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
    color_cycle=['b', 'r', 'g', 'c', 'k', 'y', 'm']
    ax.set_xscale('log')
    ax.set_yscale('log')
    realV_vs_FVMC=numpy.zeros(len(myExp_Params.regCoefs))
    diff_V_dpVL_smoothed=numpy.zeros(len(myExp_Params.regCoefs))
    diff_V_dpVL_GS=numpy.zeros(len(myExp_Params.regCoefs))
    regCoefVals=numpy.zeros(len(myExp_Params.regCoefs))
    i=0
    #for i in range(len(myExp_Params.experimentBatchLenghts)):
    m=0
    for m in range(len(myExp_Params.pow_exp)):
        diff_V_dpVL_smoothed=numpy.zeros(len(myExp_Params.regCoefs))
        diff_V_dpVL_GS=numpy.zeros(len(myExp_Params.regCoefs))
        regCoefVals=numpy.zeros(len(myExp_Params.regCoefs))
        for k in range(len(myExp_Params.experimentBatchLenghts)):
            for i in range(len(myExp_Params.regCoefs)):
                for j in range(myExp_Params.numRounds):
                    diff_V_dpVL_smoothed[i]+=(meta_exponenet_test_Reuslts[m][k][j][1][i][2]/myExp_Params.numRounds)
                    diff_V_dpVL_GS[i]+=(meta_exponenet_test_Reuslts[m][k][j][1][i][3]/myExp_Params.numRounds)
                    regCoefVals[i]=meta_exponenet_test_Reuslts[m][k][0][1][i][0]
            ax.plot(regCoefVals,diff_V_dpVL_smoothed,color=color_cycle[k])
            ax.plot(regCoefVals,diff_V_dpVL_GS,'r--',color=color_cycle[k])
            ax.legend(["diff_V_dpVL_smoothed, "+" m: "+str(myExp_Params.experimentBatchLenghts[k])," diff_V_dpVL_GS"],loc=1)
            min_dif=numpy.linalg.norm(diff_V_dpVL_smoothed-diff_V_dpVL_GS, -numpy.Inf)
            print(str(min_dif)+"batch size= "+str(myExp_Params.experimentBatchLenghts[k])+" c= "+str(myExp_Params.regCoefs[i])+" exponent= "+str(myExp_Params.pow_exp[m]))
        plt.xlabel("exponent= "+ str(myExp_Params.pow_exp[m]))
        plt.show()

def run_newGS_LSL_experiments(experimentList,myMDP_Params,myExp_Params,myMDP):   
    i=0
    regCoef=[0.5]
    expResults=[]
    for i in range(len(myExp_Params.experimentBatchLenghts)):
        expResults.append(experimentList[i].newGS_LSL_experiments(myExp_Params.experimentBatchLenghts[i],myMDP,myExp_Params.maxTrajLength, regCoef, myExp_Params.pow_exp))
    ax = plt.gca()
    ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
    realV_vs_LSL=[]
    Real_vs_GS_DPLSL=[]
    Real_vs_Smoothed_DPLSL=[]
    i=0
    for i in range(len(myExp_Params.experimentBatchLenghts)):
        temp1=[]
        temp2=[]
        temp3=[]
        
        for j in range(myExp_Params.numRounds):
            temp1.append(expResults[i][j][1])
            temp2.append(expResults[i][j][2])
            temp3.append(expResults[i][j][3])
        realV_vs_LSL.append(temp1)
        Real_vs_GS_DPLSL.append(temp2)
        Real_vs_Smoothed_DPLSL.append(temp3)
    
    mean_realV_vs_LSL=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    std_realV_vs_LSL=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    blm=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    bldl=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    bldu=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    for j in range(len(myExp_Params.experimentBatchLenghts)):
        mean_realV_vs_LSL[j] = numpy.average(realV_vs_LSL[j])#blm
        std_realV_vs_LSL [j] = numpy.std(realV_vs_LSL[j])#bld
        bldu[j] = math.log10(mean_realV_vs_LSL[j]+std_realV_vs_LSL [j])-math.log10(mean_realV_vs_LSL[j])
        bldl[j] = -math.log10(mean_realV_vs_LSL[j]-std_realV_vs_LSL [j])+math.log10(mean_realV_vs_LSL[j])
        blm[j] = math.log10(mean_realV_vs_LSL[j])
    
    mean_Real_vs_GSDPLSL=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    std_Real_vs_GSDPLSL=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    lsl_blm=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    lsl_bldl=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    lsl_bldu=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    for j in range(len(myExp_Params.experimentBatchLenghts)):
        mean_Real_vs_GSDPLSL[j] = numpy.average(Real_vs_GS_DPLSL[j])#lsl_blm
        std_Real_vs_GSDPLSL [j] = numpy.std(Real_vs_GS_DPLSL[j])#bld
        lsl_bldu[j] = math.log10(mean_Real_vs_GSDPLSL[j]+std_Real_vs_GSDPLSL [j])-math.log10(mean_Real_vs_GSDPLSL[j])
        lsl_bldl[j] = -math.log10(mean_Real_vs_GSDPLSL[j]-std_Real_vs_GSDPLSL [j])+math.log10(mean_Real_vs_GSDPLSL[j])
        lsl_blm[j] = math.log10(mean_Real_vs_GSDPLSL[j])
        
    mean_Real_vs_SmoothedDPLSL=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    std_Real_vs_SmoothedDPLSL=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    smoothed_lsl_blm=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    smoothed_lsl_bldl=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    smoothed_lsl_bldu=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    for j in range(len(myExp_Params.experimentBatchLenghts)):
        mean_Real_vs_SmoothedDPLSL[j] = numpy.average(Real_vs_Smoothed_DPLSL[j])
        std_Real_vs_SmoothedDPLSL [j] = numpy.std(Real_vs_Smoothed_DPLSL[j])
        smoothed_lsl_bldu[j] = math.log10(mean_Real_vs_SmoothedDPLSL[j]+std_Real_vs_SmoothedDPLSL [j])-math.log10(mean_Real_vs_SmoothedDPLSL[j])
        smoothed_lsl_bldl[j] = -math.log10(mean_Real_vs_SmoothedDPLSL[j]-std_Real_vs_SmoothedDPLSL [j])+math.log10(mean_Real_vs_SmoothedDPLSL[j])
        smoothed_lsl_blm[j] = math.log10(mean_Real_vs_SmoothedDPLSL[j])
    
    ax.errorbar(myExp_Params.experimentBatchLenghts, blm,  bldu, bldl)
    ax.errorbar(myExp_Params.experimentBatchLenghts, lsl_blm,  lsl_bldu, lsl_bldl)
    ax.errorbar(myExp_Params.experimentBatchLenghts, smoothed_lsl_blm,  smoothed_lsl_bldu, smoothed_lsl_bldl)

    plt.ylabel('W-RMSE')
    plt.xlabel('Batch-size')
    plt.legend(["Real-LSL","Real-(GS)LSL", "Real-(Smoothed)LSL" ],loc=3)
    plt.title("epsilon= "+str(myExp_Params.epsilon)+", delta= "+str(myExp_Params.delta)+", \lambda= 0.5 m^"+str(myExp_Params.pow_exp[0]))
    #ax.plot(myExp_Params.experimentBatchLenghts,realV_vs_LSL)
    #ax.plot(myExp_Params.experimentBatchLenghts,Real_vs_GS_DPLSL)
    plt.show()
    
def run_newGS_LSL_vs_SmoothLSL_experiments(experimentList,myMDP_Params,myExp_Params,myMDP):   
    i=0
    regCoef=[0.5]
    expResults=[]
    #expSmoothLSL_Resualts=[]
    for i in range(len(myExp_Params.experimentBatchLenghts)):
        expResults.append(experimentList[i].newGS_LSL_experiments(myExp_Params.experimentBatchLenghts[i],myMDP,myExp_Params.maxTrajLength, regCoef, myExp_Params.pow_exp))
    ax = plt.gca()
    ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
    realV_vs_LSL=[]
    Real_vs_DPLSL=[]
    Real_vs_Smoothed_DPLSL=[]
    i=0
    for i in range(len(myExp_Params.experimentBatchLenghts)):
        temp1=[]
        temp2=[]
        temp3=[]
        for j in range(myExp_Params.numRounds):
            temp1.append(expResults[i][j][1])
            temp2.append(expResults[i][j][2])
            temp3.append(expResults[i][j][3])
        realV_vs_LSL.append(temp1)
        Real_vs_DPLSL.append(temp2)
        Real_vs_Smoothed_DPLSL.append(temp3)
    
    mean_realV_vs_LSL=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    std_realV_vs_LSL=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    blm=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    bldl=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    bldu=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    for j in range(len(myExp_Params.experimentBatchLenghts)):
        mean_realV_vs_LSL[j] = numpy.average(realV_vs_LSL[j])#blm
        std_realV_vs_LSL [j] = numpy.std(realV_vs_LSL[j])#bld
        bldu[j] = math.log10(mean_realV_vs_LSL[j]+std_realV_vs_LSL [j])-math.log10(mean_realV_vs_LSL[j])
        bldl[j] = -math.log10(mean_realV_vs_LSL[j]-std_realV_vs_LSL [j])+math.log10(mean_realV_vs_LSL[j])
        blm[j] = math.log10(mean_realV_vs_LSL[j])
    
    mean_DPLSL_vs_Real=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    std_DPLSL_vs_Real=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    lsl_blm=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    lsl_bldl=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    lsl_bldu=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    for j in range(len(myExp_Params.experimentBatchLenghts)):
        mean_DPLSL_vs_Real[j] = numpy.average(Real_vs_DPLSL[j])#lsl_blm
        std_DPLSL_vs_Real [j] = numpy.std(Real_vs_DPLSL[j])#bld
        lsl_bldu[j] = math.log10(mean_DPLSL_vs_Real[j]+std_DPLSL_vs_Real [j])-math.log10(mean_DPLSL_vs_Real[j])
        lsl_bldl[j] = -math.log10(mean_DPLSL_vs_Real[j]-std_DPLSL_vs_Real [j])+math.log10(mean_DPLSL_vs_Real[j])
        lsl_blm[j] = math.log10(mean_DPLSL_vs_Real[j])
      
    mean_Smoothed_DPLSL_vs_Real=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    std_Smoothed_DPLSL_vs_Real=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    lsl_smoothed_blm=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    lsl_smoothed_bldl=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    lsl_smoothed_bldu=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    for j in range(len(myExp_Params.experimentBatchLenghts)):
        mean_Smoothed_DPLSL_vs_Real[j] = numpy.average(Real_vs_Smoothed_DPLSL[j])#lsl_blm
        std_Smoothed_DPLSL_vs_Real [j] = numpy.std(Real_vs_Smoothed_DPLSL[j])#bld
        lsl_smoothed_bldu[j] = math.log10(mean_Smoothed_DPLSL_vs_Real[j]+std_Smoothed_DPLSL_vs_Real [j])-math.log10(mean_Smoothed_DPLSL_vs_Real[j])
        lsl_smoothed_bldl[j] = -math.log10(mean_Smoothed_DPLSL_vs_Real[j]-std_Smoothed_DPLSL_vs_Real [j])+math.log10(mean_Smoothed_DPLSL_vs_Real[j])
        lsl_smoothed_blm[j] = math.log10(mean_Smoothed_DPLSL_vs_Real[j])
       
    ax.errorbar(myExp_Params.experimentBatchLenghts, blm,  bldu, bldl)
    ax.errorbar(myExp_Params.experimentBatchLenghts, lsl_blm,  lsl_bldu, lsl_bldl)
    ax.errorbar(myExp_Params.experimentBatchLenghts, lsl_smoothed_blm,  lsl_smoothed_bldu, lsl_smoothed_bldl)

    plt.xscale('log')
    #plt.yscale('log')
    #plt.ylim((-10,10))
    plt.ylabel('(log)W-RMSE')
    plt.xlabel('(log)m')
    plt.legend(["True vs. LSL","True vs. GS-DPLSL","True vs. Smoothed-DPLSL" ],loc=7)
    plt.title("epsilon= "+str(myExp_Params.epsilon)+", delta= "+str(myExp_Params.delta)+", \lambda= 0.5 m^"+str(myExp_Params.pow_exp[0]))
    #ax.plot(myExp_Params.experimentBatchLenghts,realV_vs_LSL)
    #ax.plot(myExp_Params.experimentBatchLenghts,Real_vs_DPLSL)
    plt.show()

def run_lstdExperiment(myMDP_Params, myExp_Params,myMDP,lambda_coef):
    batchSize=100
    policy="uniform"
    lambdaClass='L'
    stateSpace=myMDP.getStateSpace()
    myExp= experiment(myExp_Params.aggregationFactor,stateSpace,myExp_Params.epsilon, myExp_Params.delta, lambdaClass, myExp_Params.numRounds, batchSize,policy)
    myMCPE=MCPE(myMDP,myExp.getPhi(),myExp.getPolicy())
    data=myMCPE.batchGen(myMDP, 200, batchSize, myMDP.getGamma(), myExp.getPolicy())
    for i in range(batchSize):
        theta_hat=myMCPE.LSTD_lambda(myExp.getPhi(), lambda_coef, myMDP, data[i])
        print(theta_hat)

def SALSW_numSubs_experimet(experimentList, myMCPE,myMDP_Params, myExp_Params, myMDP,exp, subSampleSize):
    expResultsDPSA=[]
    expResultsSA=[]
    expResultsLSW=[]
    expResultsV=[]
    expResultsDPLSW=[]
    for i in range(len(myExp_Params.experimentBatchLenghts)):
        numberOfsubSamples=int((myExp_Params.experimentBatchLenghts[i])**exp)
        s=int(numpy.sqrt(numberOfsubSamples))
        tempSAE=experimentList[i].LSW_subSampleAggregateExperiment(myMDP,myExp_Params.experimentBatchLenghts[i],myExp_Params.maxTrajLength,numberOfsubSamples,s,myExp_Params.epsilon,myExp_Params.delta,experimentList[0].getPhi(),subSampleSize)
        expResultsDPSA.append(tempSAE[0])
        expResultsSA.append(tempSAE[1])
        expResultsLSW.append(tempSAE[2])
        expResultsV.append(tempSAE[3])
        expResultsDPLSW.append(tempSAE[4])
    #ax = plt.gca()
    #ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
    mean_V_vs_DPSA=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    std_V_vs_DPSA=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_DPSA_bldu=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_DPSA_bldl=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_DPSA_blm=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    mean_V_vs_SA=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    std_V_vs_SA=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_SA_bldu=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_SA_bldl=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_SA_blm=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    mean_V_vs_LSW=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    std_V_vs_LSW=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_LSW_bldu=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_LSW_bldl=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_LSW_blm=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    mean_V_vs_DPLSW=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    std_V_vs_DPLSW=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_DPLSW_bldu=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_DPLSW_bldl=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_DPLSW_blm=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    dim=len(experimentList[0].getPhi())
    for j in range(len(myExp_Params.experimentBatchLenghts)):
        tempDPSA= [[] for x in range(len(myExp_Params.experimentBatchLenghts))]
        tempSA= [[] for x in range(len(myExp_Params.experimentBatchLenghts))]
        tempV=numpy.reshape(expResultsV[j],(len(experimentList[i].getPhi()),1))
        tempLSW=[[] for x in range(len(myExp_Params.experimentBatchLenghts))]
        tempDPLSW=[[] for x in range(len(myExp_Params.experimentBatchLenghts))]
        
        for k in range(myExp_Params.numRounds):
            tempDPSA[j].append(myMCPE.weighted_dif_L2_norm(myMDP, tempV ,numpy.reshape(expResultsDPSA[j][k],(dim,1))))
            tempSA[j].append(myMCPE.weighted_dif_L2_norm(myMDP, tempV, numpy.reshape(expResultsSA[j][k],(dim,1))))
            vhat=numpy.reshape(expResultsLSW[j][k],(dim,1))
            tempLSW[j].append(myMCPE.weighted_dif_L2_norm(myMDP, tempV, vhat))
            vhatDPLSW=numpy.reshape(expResultsDPLSW[j][k],(dim,1))
            tempDPLSW[j].append(myMCPE.weighted_dif_L2_norm(myMDP, tempV, vhatDPLSW))
        temptemp=tempLSW[j]
        mean_V_vs_LSW[j]=abs(numpy.average(temptemp))
        std_V_vs_LSW[j] = numpy.std(temptemp)
        V_vs_LSW_bldu[j] = math.log10(abs(mean_V_vs_LSW[j]+std_V_vs_LSW[j]))-math.log10(abs(mean_V_vs_LSW[j]))
        V_vs_LSW_bldl[j] = -math.log10(abs(mean_V_vs_LSW[j]-std_V_vs_LSW[j]))+math.log10(abs(mean_V_vs_LSW[j]))
        V_vs_LSW_blm[j] = math.log10(abs(mean_V_vs_LSW[j]))
        
        
        mean_V_vs_DPLSW[j]=abs(numpy.average(tempDPLSW[j]))
        std_V_vs_DPLSW[j] = numpy.std(tempDPLSW[j])
        V_vs_DPLSW_bldu[j] = math.log10(abs(mean_V_vs_DPLSW[j]+std_V_vs_DPLSW[j]))-math.log10(abs(mean_V_vs_DPLSW[j]))
        V_vs_DPLSW_bldl[j] = -math.log10(abs(mean_V_vs_DPLSW[j]-std_V_vs_DPLSW[j]))+math.log10(abs(mean_V_vs_DPLSW[j]))
        V_vs_DPLSW_blm[j] = math.log10(abs(mean_V_vs_DPLSW[j]))
        
        mean_V_vs_DPSA[j]=numpy.average(tempDPSA[j])
        std_V_vs_DPSA[j] = numpy.std(tempDPSA[j])#bld
        V_vs_DPSA_bldu[j] = math.log10(abs(mean_V_vs_DPSA[j]+std_V_vs_DPSA[j]))-math.log10(abs(mean_V_vs_DPSA[j]))
        V_vs_DPSA_bldl[j] = -math.log10(abs(mean_V_vs_DPSA[j]-std_V_vs_DPSA[j]))+math.log10(abs(mean_V_vs_DPSA[j]))
        V_vs_DPSA_blm[j] = math.log10(abs(mean_V_vs_DPSA[j]))
        
        mean_V_vs_SA[j]=numpy.average(tempSA[j])
        std_V_vs_SA[j] = numpy.std(tempSA[j])#bld
        V_vs_SA_bldu[j] = math.log10((mean_V_vs_SA[j]+std_V_vs_SA[j]))-math.log10((mean_V_vs_SA[j]))
        V_vs_SA_bldl[j] = -math.log10((mean_V_vs_SA[j]-std_V_vs_SA[j]))+math.log10((mean_V_vs_SA[j]))
        V_vs_SA_blm[j] = math.log10((mean_V_vs_SA[j]))
    #ax.errorbar(myExp_Params.experimentBatchLenghts, V_vs_LSW_blm,  yerr=[V_vs_LSW_bldu, V_vs_LSW_bldl])
    #ax.errorbar(myExp_Params.experimentBatchLenghts, V_vs_DPLSW_blm,  yerr=[V_vs_DPLSW_bldu, V_vs_DPLSW_bldl])
    #ax.errorbar(myExp_Params.experimentBatchLenghts, V_vs_SA_blm,  yerr=[V_vs_SA_bldu, V_vs_SA_bldl])
    #ax.errorbar(myExp_Params.experimentBatchLenghts, V_vs_DPSA_blm,  yerr=[V_vs_DPSA_bldu, V_vs_DPSA_bldl])
    return [tempLSW[j], tempSA[j],mean_V_vs_LSW[j],mean_V_vs_SA[j]]

    #ax.set_xscale('log')
    #plt.ylabel('(log) RMSE)')
    #plt.xlabel('(log) Batch Size')
    #plt.legend(["LSW-Real", "DPLSW-Real", "(LSW)SA-Real", "(LSW)DPSA-Real"],loc=10)
    #plt.legend(["LSW-Real", "SALSW-Real"],loc=10)
    #plt.title("epsilon= "+str(myExp_Params.epsilon)+", delta= "+str(myExp_Params.delta)+", number of sub samples: "+str(numberOfsubSamples)+ " Aggregation Factor= "+str(myExp_Params.aggregationFactor))
    #ax.plot(myExp_Params.experimentBatchLenghts,realV_vs_FVMC)
    #ax.plot(myExp_Params.experimentBatchLenghts,LSL_vs_DPLSL)
    plt.show()
    
def run_SALSW_numSubs_experimet(experimentList, myMCPE,myMDP_Params, myExp_Params, myMDP,subSampleSize):
    exps=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
     
    resultsLSW=[]
    reultsSA=[]
    for exp in exps:
        resultsLSW.append(SALSW_numSubs_experimet(experimentList, myMCPE,myMDP_Params, myExp_Params, myMDP,exp)[2],subSampleSize)
        reultsSA.append(SALSW_numSubs_experimet(experimentList, myMCPE,myMDP_Params, myExp_Params, myMDP,exp)[3],subSampleSize)
    ax = plt.gca()
    ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
    #for i in len(exps):
    ax.plot(exps, resultsLSW)
    ax.plot(exps, reultsSA)
    #ax.set_xscale('log')
    plt.ylabel('(log) RMSE)')
    plt.xlabel('(log) Bucket Size')
    #plt.legend(["LSW-Real", "DPLSW-Real", "(LSW)SA-Real", "(LSW)DPSA-Real"],loc=10)
    plt.legend(["LSW-Real", "SALSW-Real"],loc=10)
    #plt.title("Batch Size= "+str(myExp_Params.experimentBatchLenghts[0])+ ",  Aggregation Factor= "+str(myExp_Params.aggregationFactor))
    plt.title("Batch Size= "+str(myExp_Params.experimentBatchLenghts[0])+ ",  Radial Basis")
    #ax.plot(myExp_Params.experimentBatchLenghts,realV_vs_FVMC)
    #ax.plot(myExp_Params.experimentBatchLenghts,LSL_vs_DPLSL)
    plt.show()
    
    
def run_LSW_SubSampAggExperiment(experimentList, myMCPE,myMDP_Params, myExp_Params, myMDP):
    
    expResultsDPSA=[]
    expResultsSA=[]
    expResultsLSW=[]
    expResultsV=[]
    expResultsDPLSW=[]
    #numberOfsubSamples=1
    subSampleSize=math.floor((experimentList[0].getBatchSize())**(3.0/4.0))
    for i in range(len(myExp_Params.experimentBatchLenghts)):
        numberOfsubSamples=int(math.pow(myExp_Params.experimentBatchLenghts[i],0.8))
        tempSAE=experimentList[i].LSW_subSampleAggregateExperiment(myMDP,myExp_Params.experimentBatchLenghts[i],myExp_Params.maxTrajLength,numberOfsubSamples,myExp_Params.epsilon,myExp_Params.delta, myExp_Params.delta_prime,experimentList[0].getPhi(),subSampleSize)
        expResultsDPSA.append(tempSAE[0])
        expResultsSA.append(tempSAE[1])
        expResultsLSW.append(tempSAE[2])
        expResultsV.append(tempSAE[3])
        expResultsDPLSW.append(tempSAE[4])
    ax = plt.gca()
    ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
    
    
    
    mean_V_vs_DPSA=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    std_V_vs_DPSA=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_DPSA_bldu=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_DPSA_bldl=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_DPSA_blm=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    mean_V_vs_SA=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    std_V_vs_SA=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_SA_bldu=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_SA_bldl=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_SA_blm=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    mean_V_vs_LSW=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    std_V_vs_LSW=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_LSW_bldu=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_LSW_bldl=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_LSW_blm=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    mean_V_vs_DPLSW=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    std_V_vs_DPLSW=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_DPLSW_bldu=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_DPLSW_bldl=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_DPLSW_blm=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    dim=len(experimentList[0].getPhi())
    
    for j in range(len(myExp_Params.experimentBatchLenghts)):
        tempDPSA= [[] for x in range(len(myExp_Params.experimentBatchLenghts))]
        tempSA= [[] for x in range(len(myExp_Params.experimentBatchLenghts))]
        tempV=numpy.reshape(expResultsV[j],(len(experimentList[i].getPhi()),1))
        tempLSW=[[] for x in range(len(myExp_Params.experimentBatchLenghts))]
        tempDPLSW=[[] for x in range(len(myExp_Params.experimentBatchLenghts))]
        
        for k in range(myExp_Params.numRounds):
            tempDPSA[j].append(myMCPE.weighted_dif_L2_norm(myMDP, tempV ,numpy.reshape(expResultsDPSA[j][k],(dim,1))))
            tempSA[j].append(myMCPE.weighted_dif_L2_norm(myMDP, tempV, numpy.reshape(expResultsSA[j][k],(dim,1))))
            vhat=numpy.reshape(expResultsLSW[j][k],(dim,1))
            tempLSW[j].append(myMCPE.weighted_dif_L2_norm(myMDP, tempV, vhat))
            vhatDPLSW=numpy.reshape(expResultsDPLSW[j][k],(dim,1))
            tempDPLSW[j].append(myMCPE.weighted_dif_L2_norm(myMDP, tempV, vhatDPLSW))
            
        #tempDPSA=tempDPSA/myExp_Params.numRounds
        #tempSA=tempSA/myExp_Params.numRounds
        #tempLSW=tempLSW/myExp_Params.numRounds
        
        
            
        #mean_V_vs_LSW[j]=numpy.average(tempLSW-tempV)
        #std_V_vs_LSW[j] = numpy.std(tempLSW-tempV)#bld
        #V_vs_LSW_bldu[j] = math.log10(abs(mean_V_vs_LSW[j]+std_V_vs_LSW[j]))-math.log10(abs(mean_V_vs_LSW[j]))
        #V_vs_LSW_bldl[j] = -math.log10(abs(mean_V_vs_LSW[j]-std_V_vs_LSW[j]))+math.log10(abs(mean_V_vs_LSW[j]))
        #V_vs_LSW_blm[j] = math.log10(abs(mean_V_vs_LSW[j]))   
        temptemp=tempLSW[j]
        mean_V_vs_LSW[j]=abs(numpy.average(temptemp))
        std_V_vs_LSW[j] = numpy.std(temptemp)
        V_vs_LSW_bldu[j] = math.log10(abs(mean_V_vs_LSW[j]+std_V_vs_LSW[j]))-math.log10(abs(mean_V_vs_LSW[j]))
        V_vs_LSW_bldl[j] = -math.log10(abs(mean_V_vs_LSW[j]-std_V_vs_LSW[j]))+math.log10(abs(mean_V_vs_LSW[j]))
        V_vs_LSW_blm[j] = math.log10(abs(mean_V_vs_LSW[j]))
        
        
        mean_V_vs_DPLSW[j]=abs(numpy.average(tempDPLSW[j]))
        std_V_vs_DPLSW[j] = numpy.std(tempDPLSW[j])
        V_vs_DPLSW_bldu[j] = math.log10(abs(mean_V_vs_DPLSW[j]+std_V_vs_DPLSW[j]))-math.log10(abs(mean_V_vs_DPLSW[j]))
        V_vs_DPLSW_bldl[j] = -math.log10(abs(mean_V_vs_DPLSW[j]-std_V_vs_DPLSW[j]))+math.log10(abs(mean_V_vs_DPLSW[j]))
        V_vs_DPLSW_blm[j] = math.log10(abs(mean_V_vs_DPLSW[j]))
        
        mean_V_vs_DPSA[j]=numpy.average(tempDPSA[j])
        std_V_vs_DPSA[j] = numpy.std(tempDPSA[j])#bld
        V_vs_DPSA_bldu[j] = math.log10(abs(mean_V_vs_DPSA[j]+std_V_vs_DPSA[j]))-math.log10(abs(mean_V_vs_DPSA[j]))
        V_vs_DPSA_bldl[j] = -math.log10(abs(mean_V_vs_DPSA[j]-std_V_vs_DPSA[j]))+math.log10(abs(mean_V_vs_DPSA[j]))
        V_vs_DPSA_blm[j] = math.log10(abs(mean_V_vs_DPSA[j]))
        
        mean_V_vs_SA[j]=numpy.average(tempSA[j])
        std_V_vs_SA[j] = numpy.std(tempSA[j])#bld
        V_vs_SA_bldu[j] = math.log10((mean_V_vs_SA[j]+std_V_vs_SA[j]))-math.log10((mean_V_vs_SA[j]))
        V_vs_SA_bldl[j] = -math.log10((mean_V_vs_SA[j]-std_V_vs_SA[j]))+math.log10((mean_V_vs_SA[j]))
        V_vs_SA_blm[j] = math.log10((mean_V_vs_SA[j]))
        
        
        
    
    #ax.errorbar(myExp_Params.experimentBatchLenghts, V_vs_LSW_blm,  yerr=[V_vs_LSW_bldu, V_vs_LSW_bldl])
    ax.errorbar(myExp_Params.experimentBatchLenghts, V_vs_DPLSW_blm,  yerr=[V_vs_DPLSW_bldu, V_vs_DPLSW_bldl])
    #ax.errorbar(myExp_Params.experimentBatchLenghts, V_vs_SA_blm,  yerr=[V_vs_SA_bldu, V_vs_SA_bldl])
    ax.errorbar(myExp_Params.experimentBatchLenghts, V_vs_DPSA_blm,  yerr=[V_vs_DPSA_bldu, V_vs_DPSA_bldl])
    

    ax.set_xscale('log')
    plt.ylabel('(log) RMSE)')
    plt.xlabel('(log) Batch Size')
    #plt.legend(["LSW-Real", "DPLSW-Real", "(LSW)SA-Real", "(LSW)DPSA-Real"],loc=10)
    plt.legend(["DPLSW vs. True", "SA-DPLSW vs. True"],loc=1)
    #plt.title("epsilon= "+str(myExp_Params.epsilon)+", delta= "+str(myExp_Params.delta)+", number of sub samples: \sqrt(m)")
    #ax.plot(myExp_Params.experimentBatchLenghts,realV_vs_FVMC)
    #ax.plot(myExp_Params.experimentBatchLenghts,LSL_vs_DPLSL)
    plt.show()
    
def run_SubSampleAggregtate_LSL_LambdaExperiment(experimentList, myMDP, myExp_Params,myMCP,Phi):
    
    
    i=0
    expResults=[]
    for i in range(len(myExp_Params.experimentBatchLenghts)):
        n_sumbsampels=int(math.sqrt(myExp_Params.experimentBatchLenghts[i]))
        s=int(numpy.sqrt(n_sumbsampels))
        expResults.append(experimentList[i].lambdaExperiment_SA_LSL(myMDP,n_sumbsampels,myExp_Params.experimentBatchLenghts[i],myExp_Params.maxTrajLength, myExp_Params.regCoefs,myExp_Params.pow_exp,s,myExp_Params.epsilon,myExp_Params.delta,Phi))
    ax = plt.gca()
    ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    num_reidge_params=len(myExp_Params.regCoefs)*len(myExp_Params.pow_exp)

    #Real_vs_LSL_list=numpy.zeros(len(num_reidge_params))
    #Real_vs_DPLSL_list=numpy.zeros(len(num_reidge_params))
    expReal_vs_LS=numpy.zeros((len(myExp_Params.experimentBatchLenghts),num_reidge_params))
    expReal_vs_DPLSL=numpy.zeros((len(myExp_Params.experimentBatchLenghts),num_reidge_params))
    reidgeParamLsit=[]
    i=0
    #num_reidge_params=len(myExp_Params.regCoefs)*len(myExp_Params.pow_exp)
    for i in range(len(myExp_Params.experimentBatchLenghts)):
        Real_vs_LSL_list=numpy.zeros((num_reidge_params))
        Real_vs_DPLSL_list=numpy.zeros((num_reidge_params))
        for j in range(myExp_Params.numRounds):
            tempLSL=[]
            tempDPLSL=[]
            reidgeParamLsit=[]
            for k in range(num_reidge_params):
                reidgeParamLsit.append(expResults[i][j][k][0])
                tempLSL.append(expResults[i][j][k][1])
                tempDPLSL.append(expResults[i][j][k][2])
            Real_vs_LSL_list+=numpy.ravel((1/myExp_Params.numRounds)*numpy.mat(tempLSL))
            Real_vs_DPLSL_list+=numpy.ravel((1/myExp_Params.numRounds)*numpy.mat(tempDPLSL))
        expReal_vs_LS[i]=Real_vs_LSL_list
        expReal_vs_DPLSL[i]=Real_vs_DPLSL_list
        
        
        
    #ax.plot(numpy.ravel(expReal_vs_LS[0]))
    #ax.plot(numpy.ravel(expReal_vs_DPLSL[0]))
    ax.plot(numpy.ravel(expReal_vs_LS[0]))
    ax.plot(numpy.ravel(expReal_vs_DPLSL[0]))
    #print(reidgeParamLsit[numpy.argmin(numpy.ravel(expReal_vs_LS[1])-numpy.ravel(expReal_vs_DPLSL[1]))])
    ax.plot(numpy.ravel(expReal_vs_LS[1]))
    ax.plot(numpy.ravel(expReal_vs_DPLSL[1]))
    #print(reidgeParamLsit[numpy.argmin(numpy.ravel(expReal_vs_LS[2])-numpy.ravel(expReal_vs_DPLSL[3]))])
    ax.plot(numpy.ravel(expReal_vs_LS[2]))
    ax.plot(numpy.ravel(expReal_vs_DPLSL[2]))
    #print(reidgeParamLsit[numpy.argmin(numpy.ravel(expReal_vs_LS[3])-numpy.ravel(expReal_vs_DPLSL[3]))])
    #print(reidgeParamLsit[-1])
    plt.show()
    
def run_LSL_SubSampAggExperiment(experimentList, myMCPE,myMDP_Params, myExp_Params, myMDP):
    
    expResultsDPSA=[]
    expResultsSA=[]
    expResultsLSL=[]
    expResultsV=[]
    expResultsDPLSL=[]

    for i in range(len(myExp_Params.experimentBatchLenghts)):
        numberOfsubSamples=int(math.sqrt(myExp_Params.experimentBatchLenghts[i]))
        s=int(numpy.sqrt(numberOfsubSamples))
        tempSAE=experimentList[i].LSL_subSampleAggregateExperiment(myMDP,myExp_Params.lambdaCoef,myExp_Params.experimentBatchLenghts[i],myExp_Params.pow_exp,myExp_Params.maxTrajLength,numberOfsubSamples,s,myExp_Params.epsilon,myExp_Params.delta,experimentList[0].getPhi(),myExp_Params.distUB)
        expResultsDPSA.append(tempSAE[0])
        expResultsSA.append(tempSAE[1])
        expResultsLSL.append(tempSAE[2])
        expResultsV.append(tempSAE[3])
        expResultsDPLSL.append(tempSAE[4])
        
    ax1 = plt.gca()
    ax1.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)

    #ax.set_color_cycle(['b', 'r', 'g', 'y', 'k', 'c', 'm'])
    
    mean_V_vs_DPSA=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    std_V_vs_DPSA=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_DPSA_bldu=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_DPSA_bldl=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_DPSA_blm=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    mean_V_vs_SA=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    std_V_vs_SA=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_SA_bldu=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_SA_bldl=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_SA_blm=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    mean_V_vs_LSL=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    std_V_vs_LSL=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_LSL_bldu=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_LSL_bldl=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_LSL_blm=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    
    
    mean_V_vs_DPLSL=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    std_V_vs_DPLSL=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_DPLSL_bldu=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_DPLSL_bldl=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    V_vs_DPLSL_blm=numpy.zeros(len(myExp_Params.experimentBatchLenghts))
    
    dim=len(experimentList[0].getPhi())
    
    for j in range(len(myExp_Params.experimentBatchLenghts)):
        tempDPSA= [[] for x in range(len(myExp_Params.experimentBatchLenghts))]
        tempSA= [[] for x in range(len(myExp_Params.experimentBatchLenghts))]
        tempV=numpy.reshape(expResultsV[j],(myMDP_Params.numState,1))
        tempLSL=[[] for x in range(len(myExp_Params.experimentBatchLenghts))]
        tempDPLSL=[[] for x in range(len(myExp_Params.experimentBatchLenghts))]
        for k in range(myExp_Params.numRounds):
            tempDPSA[j].append(myMCPE.weighted_dif_L2_norm(myMDP, tempV ,numpy.reshape(expResultsDPSA[j][k],(dim,1))))
            tempSA[j].append(myMCPE.weighted_dif_L2_norm(myMDP, tempV, numpy.reshape(expResultsSA[j][k],(dim,1))))
            vhat=numpy.reshape(expResultsLSL[j][k],(dim,1))
            tempLSL[j].append(myMCPE.weighted_dif_L2_norm(myMDP, tempV, vhat))
            vhatDPLSL=numpy.reshape(expResultsDPLSL[j][k],(dim,1))
            tempDPLSL[j].append(myMCPE.weighted_dif_L2_norm(myMDP, tempV, vhatDPLSL))
        #tempDPSA=tempDPSA/myExp_Params.numRounds
        #tempSA=tempSA/myExp_Params.numRounds
        #tempLSW=tempLSW/myExp_Params.numRounds
        
        
            
        #mean_V_vs_LSW[j]=numpy.average(tempLSW-tempV)
        #std_V_vs_LSW[j] = numpy.std(tempLSW-tempV)#bld
        #V_vs_LSW_bldu[j] = math.log10(abs(mean_V_vs_LSW[j]+std_V_vs_LSW[j]))-math.log10(abs(mean_V_vs_LSW[j]))
        #V_vs_LSW_bldl[j] = -math.log10(abs(mean_V_vs_LSW[j]-std_V_vs_LSW[j]))+math.log10(abs(mean_V_vs_LSW[j]))
        #V_vs_LSW_blm[j] = math.log10(abs(mean_V_vs_LSW[j]))   
        mean_V_vs_LSL[j]=abs(numpy.average(tempLSL[j]))
        std_V_vs_LSL[j] = numpy.std(tempLSL[j])
        V_vs_LSL_bldu[j] = math.log10(abs(mean_V_vs_LSL[j]+std_V_vs_LSL[j]))-math.log10(abs(mean_V_vs_LSL[j]))
        V_vs_LSL_bldl[j] = (-math.log10(abs(mean_V_vs_LSL[j]-std_V_vs_LSL[j]))+math.log10(abs(mean_V_vs_LSL[j])))
        V_vs_LSL_blm[j] = math.log10(abs(mean_V_vs_LSL[j]))
        
        mean_V_vs_DPLSL[j]=numpy.average(tempDPLSL[j])
        std_V_vs_DPLSL[j] = numpy.std(tempDPLSL[j])#bld
        V_vs_DPLSL_bldu[j] = math.log10(abs(mean_V_vs_DPLSL[j]+std_V_vs_DPLSL[j]))-math.log10(abs(mean_V_vs_DPLSL[j]))
        V_vs_DPLSL_bldl[j] = (-math.log10(abs(mean_V_vs_DPLSL[j]-std_V_vs_DPLSL[j]))+math.log10(abs(mean_V_vs_DPLSL[j])))
        V_vs_DPLSL_blm[j] =math.log10(abs(mean_V_vs_DPLSL[j]))
        
        
        mean_V_vs_DPSA[j]=numpy.average(tempDPSA[j])
        std_V_vs_DPSA[j] = numpy.std(tempDPSA[j])#bld
        V_vs_DPSA_bldu[j] = math.log10(abs(mean_V_vs_DPSA[j]+std_V_vs_DPSA[j]))-math.log10(abs(mean_V_vs_DPSA[j]))
        V_vs_DPSA_bldl[j] = (-math.log10(abs(mean_V_vs_DPSA[j]-std_V_vs_DPSA[j]))+math.log10(abs(mean_V_vs_DPSA[j])))
        V_vs_DPSA_blm[j] =math.log10(abs(mean_V_vs_DPSA[j]))
        
        mean_V_vs_SA[j]=numpy.average(tempSA[j])
        std_V_vs_SA[j] = numpy.std(tempSA[j])#bld
        V_vs_SA_bldu[j] = math.log10((mean_V_vs_SA[j]+std_V_vs_SA[j]))-math.log10((mean_V_vs_SA[j]))
        V_vs_SA_bldl[j] = (-math.log10((mean_V_vs_SA[j]-std_V_vs_SA[j]))+math.log10((mean_V_vs_SA[j])))
        V_vs_SA_blm[j] = math.log10((mean_V_vs_SA[j]))
        
        
        #=======================================================================
        # mean_V_vs_DPLSL[j]=numpy.average(tempDPLSL[j])
        # std_V_vs_DPLSL[j] = numpy.std(tempDPLSL[j])
        # V_vs_DPLSL_bldu[j] = math.log10((mean_V_vs_DPLSL[j]+std_V_vs_DPLSL[j]))
        # V_vs_DPLSL_bldl[j] = math.log10(abs(mean_V_vs_DPLSL[j]-std_V_vs_DPLSL[j]))
        # V_vs_DPLSL_blm[j] =math.log10(abs(mean_V_vs_DPLSL[j]))
        #=======================================================================
        
    
    

    

    ax1.set_xscale('log')
    ax1.errorbar(myExp_Params.experimentBatchLenghts, V_vs_LSL_blm, yerr=[ V_vs_LSL_bldu, V_vs_LSL_bldl])
    ax1.legend(["LSL-Real"],loc=1)
    ax1.set_xlabel('(log)Batch Size')
    ax1.set_ylabel('(log) RMSE')
    ax1.set_title("epsilon= "+str(myExp_Params.epsilon)+", delta= "+str(myExp_Params.delta)+", number of sub samples: \sqrt(m)"+"  lambda= "+ "  lambda= "+" 10000m^0.4")
    ax1.plot()
     
    ax2.set_xscale('log')
    ax2.errorbar(myExp_Params.experimentBatchLenghts, V_vs_DPLSL_blm,  yerr=[V_vs_DPLSL_bldu, V_vs_DPLSL_bldl])
    ax2.legend(["DPLSL-Real"],loc=1)
    ax2.set_xlabel('(log)Batch Size')
    ax2.set_ylabel('(log) RMSE')
    ax2.set_title("epsilon= "+str(myExp_Params.epsilon)+", delta= "+str(myExp_Params.delta)+", number of sub samples: \sqrt(m)"+"  lambda= "+ " 10000m^0.4")
    ax2.plot()
     
    ax3.set_xscale('log')
    ax3.errorbar(myExp_Params.experimentBatchLenghts, V_vs_SA_blm,  yerr=[V_vs_SA_bldu, V_vs_SA_bldl])
    ax3.legend(["(LSL)SA-Real"],loc=1)
    ax3.set_xlabel('(log)Batch Size')
    ax3.set_ylabel('(log) RMSE')
    ax3.set_title("epsilon= "+str(myExp_Params.epsilon)+", delta= "+str(myExp_Params.delta)+", number of sub samples: \sqrt(m)"+"  lambda= "+"  lambda= "+"100m^0.5")
    ax3.plot()
    
    ax4.set_xscale('log')
    ax4.errorbar(myExp_Params.experimentBatchLenghts, V_vs_DPSA_blm,  yerr=[V_vs_DPSA_bldu, V_vs_DPSA_bldl])
    ax4.legend(["DPSA(LSL)-Real"],loc=1)
    ax4.set_xlabel('(log)Batch Size')
    ax4.set_ylabel('(log) RMSE')
    ax4.set_title("epsilon= "+str(myExp_Params.epsilon)+", delta= "+str(myExp_Params.delta)+", number of sub samples: \sqrt(m)"+"  lambda= "+"100m^0.5")
    ax4.plot()
    #plt.ylabel('l2-Norm')
    #plt.xlabel('(log)Batch Size')
#     ax1.set_xlabel('(log)Batch Size')
#     ax1.set_ylabel('(log) RMSE')
#     ax1.set_title("epsilon= "+str(myExp_Params.epsilon)+", delta= "+str(myExp_Params.delta)+", number of sub samples: \sqrt(m)"+"  lambda= "+str(myExp_Params.lambdaCoef)+"m^"+str(myExp_Params.pow_exp[0]))
#     plt.legend(["LSL-Real", "DPLSL-Real",  "SA(LSL)-Real","DPSA(LSL)-Real"],loc=1)
#     ax1.set_xlim([-5, 5])
#     plt.title("epsilon= "+str(myExp_Params.epsilon)+", delta= "+str(myExp_Params.delta)+", number of sub samples: \sqrt(m)"+"  lambda= "+str(myExp_Params.lambdaCoef)+"m^"+str(myExp_Params.pow_exp[0]))
#     ax.plot(myExp_Params.experimentBatchLenghts,realV_vs_FVMC)
#     ax.plot(myExp_Params.experimentBatchLenghts,LSL_vs_DPLSL)
    
    
    
    
    
    plt.show()
    
def main():
    #######################MDP Parameters and Experiment setup###############################
    myExp_Params=expParameters()
    myMDP_Params=mdpParameteres()
    #if the absorbing state is anything except 19 the trajectory will not terminate
    absorbingStates=[]
    absorbingStates.append(myMDP_Params.numState-1)    
    goalStates=[myMDP_Params.numState-2] 
    #i=0
    #while i < myMDP_Params.numGoalstates:
    #    sRand=numpy.random.randint(0,myMDP_Params.numState)
    #    if sRand not in goalStates:
    #        goalStates.append(sRand)
    #       i=i+1  
    #    else:
    #       continue     
    stateSpace=numpy.ones(myMDP_Params.numState)
    #To DO:Fix this part, since states should be in {0,1} 
    for i in range(myMDP_Params.numState):
        stateSpace[i]=i
    stateSpace=numpy.reshape(stateSpace, (myMDP_Params.numState,1))
    ##############################Privacy Parameters###############################
    
    ##############################MCPE Parameters##################################
    lambdaClass='L'
    policy="uniform"
    distUB=myExp_Params.distUB
    #####################Generating the feature matrix#############################

    myExps=[] 
    for k in range(len(myExp_Params.experimentBatchLenghts)): 
        myExps.append(experiment(myExp_Params.aggregationFactor,stateSpace,myExp_Params.epsilon, myExp_Params.delta, lambdaClass, myExp_Params.numRounds, myExp_Params.experimentBatchLenghts[k],policy))
    featureMatrix=myExps[0].featureProducer(myExp_Params.aggregationFactor, stateSpace)
    
    
    dim=len(featureMatrix.T)
    #Starting the MC-Chain construction
    myMDP = MChain(stateSpace, myExps[0].TransitionFunction, myExps[0].rewardfunc, goalStates, absorbingStates, myMDP_Params.gamma_factor, myMDP_Params.maxReward)
    myMCPE=MCPE(myMDP,myExps[len(myExp_Params.experimentBatchLenghts)-1].getPhi(),myExps[len(myExp_Params.experimentBatchLenghts)-1].getPolicy(),"N")
    #myMCPE.batchGen(myMDP, myExp_Params.maxTrajLength, 10, myMDP_Params.gamma_factor)
    #Weight vector is used for averaging
    weightVector=[]
    for i in range(myMDP_Params.numState):
        if i==absorbingStates[0]:
            weightVector.append(0)
        else:
            weightVector.append(1/(myMDP_Params.numState-myMDP_Params.numAbsorbingStates))
    weightVector=numpy.reshape(weightVector,(myMDP_Params.numState,1))
    #run_lambdaExperiment_LSL(myExps, myMDP_Params, myExp_Params, myMDP)
    #run_newGS_LSL_experiments(myExps, myMDP_Params, myExp_Params, myMDP)
    #run_newGS_LSL_vs_SmoothLSL_experiments(myExps, myMDP_Params, myExp_Params, myMDP)
    #run_lambdaExperiment_GS_LSL(myMCPE, myExps,myMDP_Params, myExp_Params, myMDP)
    run_LSW_SubSampAggExperiment(myExps, myMCPE, myMDP_Params, myExp_Params, myMDP)
    #run_SALSW_numSubs_experimet(myExps, myMCPE, myMDP_Params, myExp_Params, myMDP)
    #run_LSL_SubSampAggExperiment(myExps, myMCPE, myMDP_Params, myExp_Params, myMDP)
    #run_SubSampleAggregtate_LSL_LambdaExperiment(myExps, myMDP, myExp_Params,myMCPE,featureMatrix)
    #print(myMCPE.computeLambdas(myMDP, featureMatrix, myExp_Params.regCoefs, 1000, myExp_Params.pow_exp)[26])  
    #run_lstdExperiment(myMDP_Params, myExp_Params, myMDP, 0.5)
    
if __name__ == "__main__": main()