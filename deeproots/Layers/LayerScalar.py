##Created : Anil Variyar
#last Modified : Anil Variyar | 12-02-2025
#Copyright : Anil Variyar | 2024 - 
#Terms of Use : MIT License | Read Terms and Conditions in the License file 
#No warranty of any kind is provided. Use at your own risk.

import numpy as np

from deeproots.Neurons.Neuron import Neuron

class LayerScalar:

    def __init__(self, nInputs, nNeurons, neuron='tanh'):
        print("Initializing a NN Layer with a tanh activation function")
        self.neurons  = [Neuron(nInputs) for i in range(nNeurons)]
        self.nNeurons = nNeurons
        self.nParams  = (nInputs+1) * nNeurons
        self.nInputs  = nInputs
        
        self.a        = np.zeros(nNeurons)
        self.x        = np.zeros(nInputs)

        self.da_dw    = np.zeros([nNeurons, nInputs+1])
        self.da_dx    = np.zeros([nNeurons, nInputs])

        self.dl_dx    = np.zeros(nInputs)
        self.dl_dw    = np.zeros([nNeurons, nInputs+1])
        
        #self.dl_dw    = np.zeros([nInputs])
        #self.dl_da    = np.zeros(nInputs)

    def update_weights(self,w):
        for i in range(0, self.nNeurons):
            self.neurons[i].update_weights(w[i*(self.nInputs+1):(i+1)*(self.nInputs+1)])

    def forward(self, x):
        for i in range(0, self.nNeurons):
            self.a[i] = self.neurons[i].forward(x)
        return self.a

    def backward(self, dL_da):

        self.dl_dx    = np.zeros(self.nInputs)
        
        for i in range(0, self.nNeurons):
            #for each neuron pass the relevant 

            #dL_dx1 -  store the dL_dx1 all together 
            dl_dx, dl_dw = self.neurons[i].backward(dL_da[i])

            #
            self.dl_dx[:] = self.dl_dx + dl_dx

            self.dl_dw[i,:] = dl_dw[:]
    

        
        return self.dl_dx


    def zero_grad(self):
        self.dl_dx = np.zeros(self.nInputs)
        self.dl_dw = np.zeros([self.nNeurons, self.nInputs+1])


    def get_gradient(self):

        return self.layers[iL].dl_dw.flatten()



    def visualize(self):
        print("Visualize Neuron")





#----------------------------------------------------------------------



def main():

    L1 = LayerScalar(2, 2, 'tanh')

    #Neuron weights
    L1.neurons[0].tag = 'N1'
    L1.neurons[0].update_weights(np.array([0.1, 0.3, 0.1]))

    L1.neurons[1].tag = 'N2'
    L1.neurons[1].update_weights(np.array([0.2, 0.2, 0.2]))


    #Forward
    x = np.array([0.2, 0.1])
    a1 = L1.forward(x)
    print("a1 ", a1)


    dl_dy = np.ones(2)

    dl1_dy = L1.backward(dl_dy)
    print("dl1 dy : ",dl1_dy)
    print("dl1 ", L1.dl_dw)


#----------------------------------------------------------------------

if __name__ == "__main__":
    main()


#----------------------------------------------------------------------

        
