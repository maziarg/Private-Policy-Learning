'''
Created on Mar 3, 2016

@author: mgomrokchi
'''

class MyClass(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        self.__initDist="uniform"
        self.__numState= 20
        self.__numAbsorbingStates=1
        self.__gamma=0.9
        self.__maxReward=1