#Created : Anil Variyar
#last Modified : Anil Variyar | 12-02-2025
#Copyright : Anil Variyar | 2024 - 
#Terms of Use : MIT License | Read Terms and Conditions in the License file 
#No warranty of any kind is provided. Use at your own risk.
#This is a simple email classifier using a neural network

import numpy as np
import json

from deeproots.Neurons import ReLU
from deeproots.NN import NNScalar
from deeproots.Layers.LayerScalar import LayerScalar
from deeproots.Optimizers.Optimizer import Optimizer
from deeproots.LossFunctions.SoftmaxCrossEntropy import SoftmaxCrossEntropy
from deeproots.Utils.OneHotEncoding import one_hot_encoding
from sentence_transformers import SentenceTransformer



#Load Data
with open('../datasets/SpamData.json', 'r') as file:
    data = json.load(file)

sentences = [d['text'] for d in data]
classes = [d['class'] for d in data]

# Text to vector embedding
st = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = st.encode(sentences)

print(len(embeddings))
x_train  = np.array(embeddings)


#one hot encoding of labels
class_list = ["Spam", "Social Media", "Work", "Marketing", "Important", "Personal"]
y_train  = one_hot_encoding(classes, class_list)

print(y_train)




#Setup you Neural Network
#Layer 1
L1 = LayerScalar(384, 50, 'tanh')
L2 = LayerScalar(50, 6, 'tanh')

layer_list = [L1, L2]
net = NNScalar(layer_list)


#Loss Function
lossfun = SoftmaxCrossEntropy()


#Data Loaders
x_act = x_train[0,:]
y_act = y_train[0,:]




# Evaluate the Network
y_pred      = net.forward(x_act)
loss, dL_dy = lossfun.eval(y_act, y_pred)
net.backward(dL_dy)
grad = net.get_gradient()



#Training the Model using Gradient Descent
weights = net.get_weights()
print("weights ", weights)


init_variables = net.get_parameters()
nVars = len(init_variables)

#optimizer ------

opti = Optimizer('gradient_descent')    

opti.set_variables(nVars, values=init_variables)

opti.set_model(net, lossfun)


epochs        = 10
#batch_size    = 10
learning_rate = 0.05

dataset = {
    'x': x_train,
    'y': y_train
}


#Run the optimizer
solution = opti.run(dataset, epochs, learning_rate)



#Run Trained model
net.update_weights(solution['w_final'])

x_test = x_train[0,:]

#Run on test set
y_pred = net.forward(x_test)
print(y_pred)

net.export_model('email_classifier_init.json')


