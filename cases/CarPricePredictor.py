##Created : Anil Variyar
#last Modified : Anil Variyar | 12-02-2025
#Copyright : Anil Variyar | 2024 - 
#Terms of Use : MIT License | Read Terms and Conditions in the License file 
#No warranty of any kind is provided. Use at your own risk.
#This is a simple email classifier using a neural network

#This is a simple car price predictor using a neuron
import numpy as np

from deeproots.Neurons.Neuron import Neuron
#from deeproots.Optimizer import GradientDescent as GD


model = Neuron(2, 'tanh', 'N1')
model.forward(np.array([1,2]))
model.backward(np.array([1]))
model.print()

#opti = GD()
#opti.model = model
#opti.params = {}

#opti.run()

#opti.viz()


