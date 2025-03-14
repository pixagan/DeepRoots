##Created : Anil Variyar
#last Modified : Anil Variyar | 12-02-2025
#Copyright : Anil Variyar | 2024 - 
#Terms of Use : MIT License | Read Terms and Conditions in the License file 
#No warranty of any kind is provided. Use at your own risk.


import numpy as np

class Tanh:

    def __init__(self):
        self.tag = 'tanh Activation'

        self.x = None
        self.a = None
    
    def forward(self,z):
    
        a = np.tanh(z)
        self.a = a

        return a

    def backward(self, dl_da):

        da_dz =  (1.0 - self.a**2.0)

        return da_dz


#----------------------------------------------------------------------



def main():

    t = Tanh()


#----------------------------------------------------------------------

if __name__ == "__main__":
    main()


#----------------------------------------------------------------------

        
