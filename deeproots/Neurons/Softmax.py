##Created : Anil Variyar
#last Modified : Anil Variyar | 12-02-2025
#Copyright : Anil Variyar | 2024 - 
#Terms of Use : MIT License | Read Terms and Conditions in the License file 
#No warranty of any kind is provided. Use at your own risk.

import numpy as np

class Softmax:

    def __init__(self):
        self.tag = 'Softmax Activation'

        self.x = None
        self.a = None
    
    def forward(self,z):
    
        num = np.exp(x)
        den = np.sum(num)

        self.a = num/den

        return self.a

    def backward(self, dl_da):

        da_dz = np.zeros([self.nInputs, self.nInputs])

        da_dz = np.outer(-self.a, self.a)
        np.fill_diagonal(da_dz, self.a * (1 - self.a))

        dl_dz = np.dot(da_dz, dl_da)

        self.dl_dz = dl_dz

        return dl_dz



#----------------------------------------------------------------------



def main():

    s = Softmax()


#----------------------------------------------------------------------

if __name__ == "__main__":
    main()


#----------------------------------------------------------------------
