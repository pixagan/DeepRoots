##Created : Anil Variyar
#last Modified : Anil Variyar | 12-02-2025
#Copyright : Anil Variyar | 2024 - 
#Terms of Use : MIT License | Read Terms and Conditions in the License file 
#No warranty of any kind is provided. Use at your own risk.


#This is a simple car price predictor using a neuron
import numpy as np

from deeproots.Dataloaders.CSVLoader import CSVLoader
from deeproots.Neurons.Neuron import Neuron
#from deeproots.Optimizer import GradientDescent as GD
from deeproots.Optimizer.Optimizer import Optimizer
from deeproots.LossFunctions.MSELoss import MSELoss


#Data Loading
df = CSVLoader().load('../datasets/carprice_data.csv')

#Data Loading and Cleaning
data_raw = df.astype(float).values

data_raw[:,0] = data_raw[:,0]/500.0 #hp
data_raw[:,1] = data_raw[:,1]/30.0 #mileage
data_raw[:,2] = data_raw[:,2]/30000.0 #price

data_len = len(data_raw)

nTrain = int(0.8*data_len)
print("nTraining Points",nTrain)

x_train = np.array(data_raw[0:nTrain, 0:2])
y_train  = np.array(data_raw[0:nTrain, 2])

x_test = np.array(data_raw[nTrain:data_len, 0:2])
y_test = np.array(data_raw[nTrain:data_len, 2])



#NN Model
model = Neuron(2, 'tanh', 'N1')
model.forward(np.array([1,2]))
model.backward(np.array([1]))
model.print()



#loss function
lossfun = MSELoss()



init_variables = model.get_parameters()
nVars = len(init_variables)

#optimizer
opti = Optimizer('gradient_descent')    
# opti.set_parameters({'learning_rate': 0.01, 
#                         'max_iter': 100, 
#                         'tol': 1e-6, 
#                         'batch_size': 10})

opti.set_variables(nVars, values=init_variables)

opti.set_model(model, lossfun)


epochs = 1000
batch_size = 10
learning_rate = 0.01

dataset = {
    'x': x_train,
    'y': y_train
}


opti.run(dataset, batch_size, epochs, learning_rate)

#opti.visualize()
#opti.export_model('CarPriceOptimizer.json')



