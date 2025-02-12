##Created : Anil Variyar
#last Modified : Anil Variyar | 12-02-2025
#Copyright : Anil Variyar | 2024 - 
#Terms of Use : MIT License | Read Terms and Conditions in the License file 
#No warranty of any kind is provided. Use at your own risk.


import numpy as np

class Sigmoid:

    def __init__(self):
        self.tag = 'Sigmoid Activation'

        self.x = None
        self.a = None
    
    def forward(self,z):
    
        a = 1.0/(1.0 + np.exp(-z))
        self.a = a

        return a

    def backward(self, dl_da):

        da_dz = self.a*(1.0 - self.a)

        return da_dz



#----------------------------------------------------------------------



def main():

    s = Sigmoid()


#----------------------------------------------------------------------

if __name__ == "__main__":
    main()


#----------------------------------------------------------------------
