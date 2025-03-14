##Created : Anil Variyar
#last Modified : Anil Variyar | 12-02-2025
#Copyright : Anil Variyar | 2024 - 
#Terms of Use : MIT License | Read Terms and Conditions in the License file 
#No warranty of any kind is provided. Use at your own risk.

import numpy as np

from deeproots.Neurons.ReLU import ReLU
from deeproots.Neurons.Sigmoid import Sigmoid
from deeproots.Neurons.Tanh import Tanh
from deeproots.Neurons.Softmax import Softmax

class Neuron:

    def __init__(self, nInputs, activation='tanh', tag='', ):
        #print("Initialize Neuron ", tag)
        self.tag = tag
        self.nInputs = nInputs
        self.w = np.random.rand(nInputs)
        
        self.b = np.random.rand(1)

        self.activation = None

        self.x = None
        self.a = None
        self.z = None
        self.dz_dw = None
        self.dz_dx = None
        
        self.da_dw = None
        self.da_dx = None

        self.dl_dw = None
        self.dl_dx = None

        self.set_activation(activation)


    def set_activation(self, activation):

        if(activation == 'tanh'):
            self.activation = Tanh()
        elif(activation == 'relu'):
            self.activation = ReLU()
        elif(activation == 'sigmoid'):
            self.activation = Sigmoid()
            

    def update_weights(self, w):
        self.w = np.array(w[0:-1])
        self.b = np.array(w[-1])


    def zero_grad(self):
        self.dz_dw = None
        self.dz_dx = None
        
        self.da_dw = None
        self.da_dx = None
        

    def get_parameters(self):
        params = np.append(self.w, self.b)
        return params


    def get_weights(self):
        params = np.append(self.w, self.b)

        return params


    def get_gradient(self):
        return np.append(self.dl_dw, self.dl_db)

    
    def clear(self):
        self.x = None
        self.a = None
        self.z = None

        self.dz_dw = None
        self.dz_dx = None
        
        self.da_dw = None
        self.da_dx = None

        self.dl_dw = None
        self.dl_dx = None

    
    def forward(self,x):
        
        z = np.dot(self.w,x) + self.b
        a  = self.activation.forward(z)
       
        
        self.x = x
        self.z = z
        self.a = a

        
        
        return a

    def backward(self, dl_da):
        
        da_dz = self.activation.backward(dl_da)
    
        dl_dz = dl_da*da_dz

        dz_dw = self.x
        dz_dx = self.w
        dz_db = 1.0

        dl_dw = dl_dz*dz_dw
        dl_db = dl_dz*dz_db
        dl_dx = dl_dz*dz_dx
        

        self.dl_da = dl_da
        self.dl_dz = dl_dz
        self.dl_dw = dl_dw
        self.dl_db = dl_db
        self.dl_dx = dl_dx

        self.da_dz = da_dz
        self.dz_dw = dz_dw
        self.dz_dx = dz_dx
        self.dz_db = dz_db

        dl_dparams = np.append(dl_dw, dl_db)

        return dl_dx, dl_dparams


    def print(self):

        print("dL_da : ", self.dl_da)
        
        print("dz_dw : ", self.dz_dw)
        print("dz_dx : ", self.dz_dx)

        print("da_dw : ", self.da_dw)
        print("da_dx : ", self.da_dx)

        print("-----------------------------------")
        print("dL_dw : ", self.dl_dw)
        print("dL_dx : ", self.dl_dx)

        print("-----------------------------------")
        print("-----------------------------------")
        print("-----------------------------------")
        

    def visualize(self):
        print("Visualize Network")



#----------------------------------------------------------------------



def main():

    n = Neuron(2, 'tanh','n1')
    n.forward(np.array([1,2]))
    n.backward(np.array([1]))
    n.print()


#----------------------------------------------------------------------

if __name__ == "__main__":
    main()


#----------------------------------------------------------------------

        
