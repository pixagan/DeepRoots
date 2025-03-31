# A light wrapper 
# To enable portability for future CPU/GPU implementations
# and to enable switching between numpy and other libraries


import numpy as np

class Mat:

    def __init__(self, nRows, nCols, tag='', init='random'):
        self.nRows = nRows
        self.nCols = nCols
        self.tag   = tag
        self.data  = None
        self.initialize(init)

    def initialize(self, init='random'):
        if init == 'random':
            self.data = np.random.randn(self.nRows, self.nCols)
        elif init == 'zeros':
            self.data = np.zeros(self.nRows, self.nCols)

    def __str__(self):

        return f"Mat({self.nRows}, {self.nCols}, {self.tag})"

    def __repr__(self):
        return self.__str__()


    def shape(self):
        return (self.nRows, self.nCols)


    def transpose(self):
        return Mat(self.nCols, self.nRows, tag=self.tag, data=self.data.T)

