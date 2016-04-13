'''
Created on Mar 3, 2016

@author: mgomrokchi
'''

class mdpParameteres(object):
    '''
    classdocs
    '''


    def __init__(self):
        self.initDist="uniform"
        self.numState= 40
        self.numAbsorbingStates=1
        self.gamma=0.9
        self.maxReward=1
        self.numGoalstates=1