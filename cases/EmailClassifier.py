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
from deeproots.Optimizer import GradientDescent as GD

#Required Optimizer and Data Loaders
#Not functional yet

#Layer 1
L1 = LayerScalar(384, 50, 'tanh')
L2 = LayerScalar(50, 6, 'tanh')

layer_list = [L1, L2]
net = NNScalar(layer_list)

#Data Loaders
#x_act = X_train[0,:]
#y_act = y_train[0,:]


y_pred      = net.forward(x_act)
loss, dL_dy = lossfun.eval(y_act, y_pred)
net.backward(dL_dy)
grad = net.get_gradient()


weights = net.get_weights()
print("weights ", weights)

#lossfun =  SoftmaxCrossEntropyLoss()

