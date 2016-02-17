import numpy
import matplotlib.pyplot as plt
'''
Created on Sep 18, 2015

@author: Lucas Lehnert (lucas.lehnert@mail.mcgill.ca)
'''
import numpy as np
#from objc._objc import NULL

class MDPTabular( object ):
    '''
    classdocs
    '''


    def __init__( self, stateSpace, actionSpace, transitionFunction, rewardFunction, \
                  goalStates,  absorbingStates, maxReward=1, gamma=1.0, startDistribution=None):
        '''
        Construct a tabular MDP class.
        
        @param stateSpace: State space.
        @param actionSpace: Action space.
        @param transitionFunction: Transition function.
        @param rewardFunction: Reward function.
        @param gamma: Discount factor.
        @param startDistribution: Start state distribution. If not specified, a unifrom
            distribution over states is assumed.
        @param goalState: Goal state of the MDP. If not specified, the MDP does not
            have a goal state.
        '''
        self.__stateSpace = np.array( stateSpace, copy=True )
        self.__stateSpace.flags.writeable = False
        self.__actionSpace = np.array( actionSpace, copy=True )
        self.__actionSpace.flags.writeable = False
        self.__gamma = gamma
        self.__absorbingStates=absorbingStates
        self.__goalStates = goalStates
        self.__maxReward=maxReward


        self.__transitionModel = self._constructTransitionMatrix( transitionFunction )
        self.__transitionModelV2=self._constructTransitionMatrixV2(transitionFunction)
        self.__rewardMatrix = self._contructRewardMatrix( rewardFunction )

        if startDistribution is None:
            #this is just producing uniform dist
            self.__startDistribution = np.ones( len( self.__stateSpace ) ) / float( len( self.__stateSpace ) )
        else:
            self.__startDistribution = np.array( startDistribution, copy=True )
        self.__startDistribution.flags.writeable = False


    def _constructTransitionMatrix( self, transitionFunction ):
        S = self.__stateSpace
        A = self.__actionSpace
        T = np.zeros( ( len( S ) * len( self.__actionSpace ), len( S ) ) )
        for si in range( len( S ) ):
            for ai in range( len( A ) ):
                for sj in range( len( S ) ):
                    if  sj in self.__absorbingStates:
                        T[ ai * len( S ) + si, sj  ] = 0
                    else:
                        T[ ai * len( S ) + si, sj  ] = transitionFunction( si, A[ai], sj )
                        
        T.flags.writeable = False
        return T
    def _constructTransitionMatrixV2( self, transitionFunction ):
        S= self.__stateSpace
        A= self.__actionSpace
        Tr = np.zeros( (len( S ) , len( S ) )) 
        for si in range( len( S ) ):
            for ai in A:    
                for sj in range( len( S ) ):
                    if  si in self.__absorbingStates:
                        Tr[ si, sj  ] = 0
                    else:
                        Tr[si, sj ] = transitionFunction( si, ai ,sj )
                        
        Tr.flags.writeable = False
        return Tr
    def getTransitionMatrix( self, action ):
        ai = self.indexOfAction( action )
        return np.array( self.__transitionModel[ai * len( self.__stateSpace ):( ai + 1 ) * len( self.__stateSpace )], copy=False )
    
    def getGeneralTransitionMatix(self):
        return self.__transitionModel

    def getNextStateDistribution( self, state, action ):
        Ta = self.getTransitionMatrix( action )
        si = self.indexOfState( state )
        ds = np.zeros( len( self.__stateSpace ) )
        ds[si] = 1.0
        nextStateDistr = np.dot( ds, Ta )
        return nextStateDistr

    def sampleNextState( self, state, action ):
        nextStateDistr = self.getNextStateDistribution( state, action )
        nextStateInd = np.argmax( np.random.multinomial( 1, nextStateDistr ) )
        nextState = self.__stateSpace[nextStateInd]
        return nextState

    def sampleStartState( self ):
        startStateInd = np.argmax( np.random.multinomial( 1, self.__startDistribution ) )
        startState = self.__stateSpace[startStateInd]
        return startState

    def startStateDistribution( self ):
        return np.array( self.__startDistribution, copy=False )

    def _contructRewardMatrix( self, rewardFunction ):
        S = self.__stateSpace
        A = self.__actionSpace
        R = np.zeros( ( len( S ) * len( self.__actionSpace ), len( S ) ) )
        for si in range( len( S ) ):
            for ai in range( len( A ) ):
                for sj in range( len( S ) ):
                    R[ ai * len( S ) + si, sj ] = rewardFunction( si, A[ai], sj, self.__goalStates, self.__maxReward)
        return R

    def getReward( self, state, action, nextState ):
        si = self.indexOfState( state )
        ai = self.indexOfAction( action )
        sj = self.indexOfState( nextState )
        return self.__rewardMatrix[ ai * len( self.__stateSpace ) + si, sj ]

    def getRewardExpected( self, state, action ):
        si = self.indexOfState( state )
        ai = self.indexOfAction( action )
        rewards = self.__rewardMatrix[ ai * len( self.__stateSpace ) + si ]
        nextStateDistr = self.getNextStateDistribution( state, action )
        return np.dot( rewards, nextStateDistr )
    
    def getExpectedRewardVec(self):
        R=[]
        for si in self.getStateSpace():
            temp=0
            for ai in self.getActionSpace():
                    temp = temp+self.getRewardExpected(si, ai)
            R.append([temp]) 
        return R       
                
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

    def indexOfStateActionPair( self, sapair ):
        '''
        Get the index of a state action pair.
        
        @param sapair: A pair (state, action)
        
        @return: Index of the pair in state-action space (as returned by getStateActionPairIterable)
        '''
        indState = self.indexOfState( sapair[0] )
        indAction = self.indexOfAction( sapair[1] )
        return indAction * len( self.getStateSpace() ) + indState

    def getStateActionPairIterable( self ):
        '''
        @return: An iterable object that iterates over all state action pairs.
        '''

        class SAIter:

            def __init__( self, stateSpace, actionSpace ):
                self.__stateSpace = stateSpace
                self.__actionSpace = actionSpace
                self.__ind = 0

            def next( self ):
                actInd = self.__ind / len( self.__stateSpace )
                stateInd = self.__ind % len( self.__stateSpace )
                self.__ind += 1
                if self.__ind > len( self.__stateSpace ) * len( self.__actionSpace ):
                    raise StopIteration
                return ( self.__stateSpace[stateInd], self.__actionSpace[actInd] )

        class SA:

            def __init__( self, stateSpace, actionSpace ):
                self.__stateSpace = stateSpace
                self.__actionSpace = actionSpace

            def __iter__( self ):
                return SAIter( self.__stateSpace, self.__actionSpace )

            def __len__( self ):
                return len( self.__stateSpace ) * len( self.__actionSpace )

            def __getitem__( self, i ):
                return self.__getslice__( i, i + 1 )[0]

            def __getslice__( self, i, j ):
                res = []
                for ind in range( i, j ):
                    actInd = ind / len( self.__stateSpace )
                    stateInd = ind % len( self.__stateSpace )
                    res.append( ( self.__stateSpace[stateInd], self.__actionSpace[actInd] ) )
                return res

        return SA( self.getStateSpace(), self.getActionSpace() )

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
    def getRewardMatrix(self):
        return self.__rewardMatrix
    



