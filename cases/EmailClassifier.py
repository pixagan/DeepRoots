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
from deeproots.Optimizer.Optimizer import Optimizer
from deeproots.LossFunctions.SoftmaxCrossEntropy import SoftmaxCrossEntropy

from sentence_transformers import SentenceTransformer
#Required Optimizer and Data Loaders
#Not functional yet

#one hot encoding for classification
def one_hot_encoding(classes_in):
    
    encoding = np.zeros([len(classes_in), 6])
    for iC in range(0, len(classes_in)):
        if(classes_in[iC] == 'Spam'):
            encoding[iC, 0] = 1

        if(classes_in[iC] == 'Social Media'):
            encoding[iC, 1] = 1

        if(classes_in[iC] == 'Work'):
            encoding[iC, 2] = 1
        
        if(classes_in[iC] == 'Marketing'):
            encoding[iC, 3] = 1
        
        if(classes_in[iC] == 'Important'):
            encoding[iC, 4] = 1
        
        if(classes_in[iC] == 'Personal'):
            encoding[iC, 5] = 1

    return encoding



#Load Data
with open('../datasets/SpamData.json', 'r') as file:
    data = json.load(file)

sentences = [d['text'] for d in data]
classes = [d['class'] for d in data]

st = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = st.encode(sentences)

print(len(embeddings))
x_train  = np.array(embeddings)
y_train = one_hot_encoding(classes)
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
batch_size    = 10
learning_rate = 0.05

dataset = {
    'x': x_train,
    'y': y_train
}


solution = opti.run(dataset, batch_size, epochs, learning_rate)



solution = opti.run(dataset, batch_size, epochs, learning_rate)

opti.visualize.loss_function()


#Run Trained model
net.update_weights(solution['w_final'])

x_test = x_train[0,:]

#Run on test set
y_pred = net.forward(x_test)
print(y_pred)

net.export_model('email_classifier_init.json')


# #Open a new net
# net_new = NNScalar()
# net_new.import_model('email_classifier.json')
# weights_new = net_new.get_weights()

# print(weights_new)

# y_pred = net_new.forward(x_test)
# print(y_pred)
