##Created : Anil Variyar
#last Modified : Anil Variyar | 12-02-2025
#Copyright : Anil Variyar | 2024 - 
#Terms of Use : MIT License | Read Terms and Conditions in the License file 
#No warranty of any kind is provided. Use at your own risk.

class MSEloss:

    def __init__(self, tag=''):
        print("Initialize Loss ", tag)
        self.tag = tag
        self.loss = 0.0
        self.dloss = 0.0


    def eval(self, y_act, y_pred):
        
        #print("Computing the Loss")
        loss = (y_act - y_pred)**2.0
    
        #dl/dy
        dloss = -2.0*(y_act - y_pred)

        self.loss  = loss
        self.dloss = dloss

        return loss, dloss
