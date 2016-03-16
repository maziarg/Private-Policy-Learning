import numpy as np
import matplotlib.pyplot as plt
'''
Created on Jan 21, 2016

@author: mgomrokchi
'''

class MChain(object):
    '''
    classdocs
    '''


    def __init__(self, stateSpace, transitionFunction, rewardFunction, \
                  goalStates,  absorbingStates, gamma, maxReward, startDistribution=None):
        '''
        Main MC parameters initialization 
        '''
        self.__stateSpace = np.array( stateSpace, copy=True )
        self.__stateSpace.flags.writeable = False
        self.__gamma = gamma
        self.__absorbingStates=absorbingStates
        self.__goalStates = goalStates
        self.__maxReward=maxReward
        self.__rewardFunc=rewardFunction
        
        
        self.__transitionModel = self._constructTransitionMatrix( transitionFunction )
        self.__rewardVec = self._constructExpectedRewardVec( rewardFunction )

        
        if startDistribution==None:
            self.__startStateDist= np.ones( len( self.__stateSpace ) ) / float( len( self.__stateSpace ) )
        else:
            self.__startStateDist = np.array( startDistribution, copy=True )
        self.__startStateDist.flags.writeable = False
    def getGoalstates(self):
        return self.__goalStates        
    def getGammaMatrix(self):  
        gammaMatrix= [[ ] for y in range(len(self.__stateSpace))]   
        for i in range(len(self.__stateSpace)):
            for j in range(len(self.__stateSpace)):
                if i is j:
                    gammaMatrix[i].append(self.__startStateDist[i])
                else:
                    gammaMatrix[i].append(0) 
        return gammaMatrix          
    def _constructTransitionMatrix( self, transitionFunction ):
        S = self.__stateSpace
        T = np.zeros( ( len( S ) , len( S ) ) )
        for si in range( len( S ) ):
            for sj in range( len( S ) ):
                if  si==self.__absorbingStates:
                    if si==sj:
                        T[ si, sj ] = 1
                    else:
                        T[ si, sj ] = 0
                else:
                    T[si, sj ] = transitionFunction( si, sj )          
        T.flags.writeable = False
        return T
        
    def getTransitionMatix(self):
        return self.__transitionModel


    def sampleStartState( self ):
        startStateInd = np.argmax( np.random.multinomial( 1, self.__startStateDist))
        startState = self.__stateSpace[startStateInd]
        return startState
        
    def startStateDistribution( self ):
        return np.array( self.__startStateDist, copy=False )
    
    def getRewardExpected( self, state, rewardfunc):
        si = self.indexOfState( state )
        nextStateDist = self.getNextStateDistribution(si)
        expRew=0
        for i in range(len(nextStateDist)):
            if i in self.__goalStates:
                expRew= expRew+nextStateDist[i]*self.__maxReward 
                
        return expRew
    def getRewardDeterministic(self, destState,rewardfunc):
        si = self.indexOfState(destState)
        detReward= rewardfunc(si, self.__goalStates, self.__maxReward)
        return detReward
        
        
        
    def getReward(self,source,dest):
        if dest in self.__goalStates:
            if source in self.__goalStates:
                return 0
            else:
                return self.__maxReward
        else:
            return 0
    
            
    def _constructExpectedRewardVec(self,rewardfunc):
        R=[]
        for si in self.getStateSpace():
            temp=0
            temp = temp+self.getRewardDeterministic(si, rewardfunc)
            R.append([temp]) 
        return R 
    def getExpextedRewardVec(self):
        return self._constructExpectedRewardVec(self.__rewardFunc)
    
    def sampleNextState( self, state):
        nextStateDistr = self.getNextStateDistribution( state )
        nextStateInd = np.argmax( np.random.multinomial( 1, nextStateDistr ) )
        nextState = self.__stateSpace[nextStateInd]
        return nextState
    
    def getNextStateDistribution( self, state):
        si = self.indexOfState( state )
        nextStateDistr = self.__transitionModel[si,:]
        return nextStateDistr
        
    def indexOfState( self, state ):
        '''
        @return: Index of state in state space.
        '''
        indAr = np.where( np.all( self.getStateSpace() == state, axis=1 ) )
        if len( indAr ) == 0:
            raise Exception( 'Given state vector is not in state space. stateVector=' + str( state ) )
        if len(indAr[0])==0:
            return None
        else: 
            return int( indAr[0] )

    def indexOfAction( self, action ):
        '''
        @return: Index of action in action space.
        '''
        return int( np.where( self.getActionSpace() == action )[0][0] )
        
    def isGoalState( self, stateVector ):
        return np.all( stateVector == self.__goalState )

    def getActionSpace( self ):
        return np.array( self.__actionSpace, copy=False )
    
    def getObsorbingStates(self):
        return np.array( self.__absorbingStates, copy=False )

    def getStateSpace( self ):
        return np.array( self.__stateSpace, copy=False )

    def getGamma( self ):
        return self.__gamma
    def getMaxReward(self):
        return self.__maxReward
    
        
        