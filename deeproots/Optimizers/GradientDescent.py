##Created : Anil Variyar
#last Modified : Anil Variyar | 12-02-2025
#Copyright : Anil Variyar | 2024 - 
#Terms of Use : MIT License | Read Terms and Conditions in the License file 
#No warranty of any kind is provided. Use at your own risk.

import numpy as np

class GradientDescent():

    def __init__(self):
        super().__init__()
        print("Initializing Gradient Descent")
        self.params   = None


    def set_parameters(self, parameters):
        self.params = parameters


    def step(self, w, dw):
    
        #update the design variables
        w = w - self.params['learning_rate']*dw

        return w







#----------------------------------------------------------------------


def main():

    opti = GradientDescent()
   


#----------------------------------------------------------------------

if __name__ == "__main__":
    main()


#----------------------------------------------------------------------

        

 