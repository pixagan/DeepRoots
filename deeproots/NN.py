##Created : Anil Variyar
#last Modified : Anil Variyar | 12-02-2025
#Copyright : Anil Variyar | 2024 - 
#Terms of Use : MIT License | Read Terms and Conditions in the License file 
#No warranty of any kind is provided. Use at your own risk.

import numpy as np
import json

from deeproots.Layers.LayerScalar import LayerScalar


#Base NN  Class
class NN:

    def __init__(self):
        print("Initializing a Neural Network")
        self.netName = ''


    def update_weights(self, w):
        pass


    def export_model(self, filename):

        net_dict = {}
        net_dict['nLayers'] = self.nLayers

        for iL in range(0, self.nLayers):
            layer_dict = {}
            layer_dict['nNeurons'] = self.layers[iL].nNeurons
            layer_dict['nInputs']  = self.layers[iL].nInputs
            layer_dict['neuron']   = self.layers[iL].neuron

            weights = []
            for iN in range(0, self.layers[iL].nNeurons):
                neuron_dict = {}

                neuron_dict['w'] = self.layers[iL].neurons[iN].w.tolist()
                neuron_dict['b'] = self.layers[iL].neurons[iN].b.tolist()
                weights.append(neuron_dict)

            layer_dict['weights'] = weights
            net_dict['layers_'+str(iL+1)] = layer_dict


        print(net_dict)

        with open(filename, 'w') as f:
            json.dump(net_dict, f)

        print("Model exported to ", filename)

        return 0
        

    def import_model(self, filename):

        net_dict = json.load(open(filename, 'r'))

        self.nLayers = net_dict['nLayers']

        for iL in range(0, self.nLayers):

            layer_dict = net_dict['layers_'+str(iL+1)]

            layer_i = LayerScalar(layer_dict['nInputs'], layer_dict['nNeurons'], layer_dict['neuron'])
            weights = []
            for iN in range(0, layer_dict['nNeurons']):
                neuron_dict = layer_dict['weights'][iN]
                w = np.array(neuron_dict['w'])
                b = np.array(neuron_dict['b'])
                weights.extend(np.append(w, b))

            layer_i.update_weights(np.array(weights))

            self.layers.append(layer_i)
            

       




#A Scalar NN, layers are assembly of Neurons
class NNScalar(NN):

    def __init__(self, layers=[]):
        print("Initializing a Neural Network with n Layers")
        self.layers = layers
        self.nLayers = len(layers)
        self.layerNeurons = []

    
    def update_weights(self,w):
        currIndex = 0
        for iL in range(0, self.nLayers):
            for iN in range(0, self.layers[iL].nNeurons):
                nParamsi = self.layers[iL].nInputs + 1
                
                self.layers[iL].neurons[iN].update_weights(w[currIndex : currIndex+ nParamsi])
                currIndex = currIndex +  nParamsi                                 
            

    def view_weights(self):
        for iL in range(0, self.nLayers):
            for iN in range(0, self.layers[iL].nNeurons):
                print(iL, iN, self.layers[iL].neurons[iN].w, self.layers[iL].neurons[iN].b)

    
    def forward(self, x_in):
        
        a_i = x_in
        for i in range(0, self.nLayers):
            ai_1 = self.layers[i].forward(a_i)
            a_i = ai_1
            
        return a_i
    
    
    def backward(self, dL):
        for i in range(0, self.nLayers):
            dL_dx = self.layers[-1-i].backward(dL)
            dL = dL_dx



    def zero_grad(self):
        for i in range(0, self.nLayers):
            self.layers[i].zero_grad()



    def get_gradient(self):

        grad = np.array([])
        for iL in range(0, self.nLayers):
            grad = np.append(grad, self.layers[iL].dl_dw.flatten())

        return grad


    def get_parameters(self):
        params = np.array([])
        for iL in range(0, self.nLayers):
            for iN in range(0, self.layers[iL].nNeurons):
                wL = self.layers[iL].neurons[iN].get_weights()
                params = np.append(params, wL)
        return params


    def get_weights(self):

        weights = np.array([])

        for iL in range(0, self.nLayers):
            for iN in range(0, self.layers[iL].nNeurons):
                wL = self.layers[iL].neurons[iN].get_weights()
                weights = np.append(weights, wL)
                
                
        return weights
    
             

    def evaluate_batch(self, mode, x_in):

        batch_out = []

        for i in range(0, len(x_in)):
            x = x_in[i]
            y = y_in[i]

            a_i = self.forward(x)

            batch_out.append(a_i)

        return batch_out

                

    
    def viz(self):
    
        print(self.layers)

        layers = self.layers
    
        dotN = Digraph(format='svg', graph_attr={'rankdir':'LR'})
        
        #generate nodes
        for iL in range(0, len(layers)):
            for iN in range(0, len(layers[iL].neurons)):
                print(layers[iL].neurons[iN].tag, str(layers[iL].neurons[iN].w), str(layers[iL].neurons[iN].dl_dw))
                dotN.node(layers[iL].neurons[iN].tag, label="{ %s | %s | %s }" %(layers[iL].neurons[iN].tag, str(np.around(layers[iL].neurons[iN].w,3)), str(np.around(layers[iL].neurons[iN].dl_dw,3))), shape='record')
    
    
        
        for iL in range(0, len(layers)-1):
            for iN in range(0, len(layers[iL].neurons)):
                for iN2 in range(0, len(layers[iL+1].neurons)):
                    dotN.edge(layers[iL].neurons[iN].tag, layers[iL+1].neurons[iN2].tag, label='a'+str(iL+1)+str(iN+1) + "| " + str(np.around(layers[iL].neurons[iN2].a,3)) + "|" + str(np.around(layers[iL].neurons[iN2].dl_da,3)) )
        
        return dotN
    

        
    




#----------------------------------------------------------------------



def main():

    #Example 2 : Backprop

    #Layer 1
    L1 = LayerScalar(2, 2, 'tanh')
    L1.neurons[0].tag = 'N1'
    L1.neurons[0].update_weights(np.array([0.1, 0.3, 0.1]))

    L1.neurons[1].tag = 'N2'
    L1.neurons[1].update_weights(np.array([0.2, 0.2, 0.2]))


    #Layer 2
    L2 = LayerScalar(2, 2, 'tanh')
    L2.neurons[0].tag = 'N3'
    L2.neurons[0].update_weights(np.array([0.5, 0.1, 0.1]))


    L2.neurons[1].tag = 'N4'
    L2.neurons[1].update_weights(np.array([0.1, 0.2, 0.3]))

    #Layer 3
    L3 = LayerScalar(2, 1, 'tanh')
    L3.neurons[0].tag = 'N5'
    L3.neurons[0].update_weights(np.array([0.1, 0.2, 0.5]))

    layer_list = [L1, L2, L3]


    net = NNScalar(layer_list)



    x = np.array([0.2, 0.1])
    net.forward(x)


    #Loss 
    dL_dy = np.ones(2)


    net.backward(dL_dy)

    print(net.get_gradient())
#----------------------------------------------------------------------

if __name__ == "__main__":
    main()


#----------------------------------------------------------------------

        
