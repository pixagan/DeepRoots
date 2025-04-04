#Created : Anil Variyar
#last Modified : Anil Variyar | 12-02-2025
#Copyright : Anil Variyar | 2024 - 
#Terms of Use : MIT License | Read Terms and Conditions in the License file 
#No warranty of any kind is provided. Use at your own risk.
#This is a simple email classifier using a neural network

import numpy as np

from deeproots.Neurons import ReLU
from deeproots.NN import NNScalar
from deeproots.Layers.LayerScalar import LayerScalar
from deeproots.LossFunctions.SoftmaxCrossEntropy import SoftmaxCrossEntropy



#Required Optimizer and Data Loaders
#Not functional yet

X_train = np.array([[1,2,3], [4,5,6]])
y_train = np.array([[1,0], [0,1]])

#Layer 1
L1 = LayerScalar(3, 4, 'tanh')
L2 = LayerScalar(4, 2, 'tanh')

layer_list = [L1, L2]
net = NNScalar(layer_list)

lossfun = SoftmaxCrossEntropy()
#Data Loaders
x_act = X_train[0,:]
y_act = y_train[0,:]

# Forward pass through the network
y_pred      = net.forward(x_act)

#Evaluating the loss function
loss, dL_dy = lossfun.eval(y_act, y_pred)

#Backward pass through the network
net.backward(dL_dy)

#Getting the gradients
grad = net.get_gradient()

#Getting the weights
weights = net.get_weights()
print("weights ", weights)



