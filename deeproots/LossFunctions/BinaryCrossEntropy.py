##Created : Anil Variyar
#last Modified : Anil Variyar | 12-02-2025
#Copyright : Anil Variyar | 2024 - 
#Terms of Use : MIT License | Read Terms and Conditions in the License file 
#No warranty of any kind is provided. Use at your own risk.

class BinaryClassificationLoss:

    def __init__(self):
        print("Binary Loss Function")
        self.loss = 0.0
        self.dloss = 0.0


    def eval(self, y, yh):

        eps = 1e-9
        
        error = -(y*np.log(yh) + (1.0- y)*np.log(1.0 - yh))

        dl_dyh = -(y - yh)/((yh + eps)*(1 - yh + eps))

        self.loss = error
        self.dloss = dl_dyh

        return error, dl_dyh
