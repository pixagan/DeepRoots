##Created : Anil Variyar
#last Modified : Anil Variyar | 12-02-2025
#Copyright : Anil Variyar | 2024 - 
#Terms of Use : MIT License | Read Terms and Conditions in the License file 
#No warranty of any kind is provided. Use at your own risk.

class NNCrossEntropyLoss:

    def __init__(self):
        print("NN Cross Entropy")
        self.loss = 0.0
        self.dloss = 0.0

    def eval(self, y_act, y_hat):
        print("Loss")

        loss = 0.0
        dloss = 0.0

        loss = np.sum(y_act*np.log(y_hat))

        dloss = -1.0*y_act/y_hat

        return loss, dloss

