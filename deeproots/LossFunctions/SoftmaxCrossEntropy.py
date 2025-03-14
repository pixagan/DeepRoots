## Created by Anil Variyar
## Last Modified : Anil Variyar | 14-03-2025
## Copyright : Anil Variyar | 2024 - 
## Terms of Use : MIT License | Read Terms and Conditions in the License file 
## No warranty of any kind is provided. Use at your own risk.

import numpy as np


class SoftmaxCrossEntropy:

    def __init__(self):
        print("Softmax Cross Entropy")


    def eval(self, y_act, y_pred):

        loss  = 0.0
        dloss = 0.0

        num = np.exp(y_pred)
        den = np.sum(num)

        a = num/den
        
        loss = -np.sum(y_act*np.log(a))

        dloss = a - y_act

        return loss, dloss
