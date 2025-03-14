##Created : Anil Variyar
#last Modified : Anil Variyar | 12-02-2025
#Copyright : Anil Variyar | 2024 - 
#Terms of Use : MIT License | Read Terms and Conditions in the License file 
#No warranty of any kind is provided. Use at your own risk.


#An outline for an optimizer
#Not yet functional

import numpy as np
import matplotlib.pyplot as plt
from deeproots.Optimizer.GradientDescent import GradientDescent


class NNModel:

    def __init__(self):
        self.w = None

    def set_neural_network(self, network, lossfun):
        self.nn = nn
        self.lossfun = lossfun


    def evaluate(self, w):

        self.nn.set_weights(w)
        self.nn.evaluate(w)

    def sens(self, w):
        
        self.nn.set_weights(w)
        yhat        = self.nn.forward(self.dvs)
        loss, dloss = self.lossfun.eval(self.model.y, self.model.y_pred)
        dl_dw       += self.model.backward(dloss)





class Optimizer:

    def __init__(self, algorithm='gradient_descent', parameters={}):
        self.model     = None
        self.params    = {}
        self.dvs       = None
        self.optimizer = None

        self.set_method(algorithm)

        self.loss_history = []
        self.w_history    = []


    def set_method(self, algorithm):

        if algorithm == 'gradient_descent':
            self.optimizer = GradientDescent()

        else:
            self.optimizer = GradientDescent()



    def set_parameters(self, parameters):

        self.params = parameters



    def set_variables(self, nVars, values=None, init_type='random'):

        if(values is None):
            self.dvs = np.random.randn(nVars)

            if(init_type == 'random'):
                self.dvs = np.random.randn(nVars)

            elif(init_type == 'uniform'):
                self.dvs = np.random.uniform(0, 1, nVars)

            elif(init_type == 'ones'):
                self.dvs = np.ones(nVars)

            elif(init_type == 'zeros'):
                self.dvs = np.zeros(nVars)
            
            else:
                print("Invalid initialization type, Initializing with random values")
                self.dvs = np.random.randn(nVars)

        else:
            self.dvs = np.array(values)



    def step(self, iter):
        iter = iter + 1



    def set_model(self, model, lossfun):

        self.model = model
        self.lossfun = lossfun



    def reset_optimization(self):

        self.loss_history = []
        self.w_history    = []



    def run(self, dataset, batch_size, epochs=100, learning_rate=0.001):

        w     = self.dvs
        nVars = len(w)
        #batch_size = self.params['batch_size']
        #max_iter   = self.params['max_iter']

        x = dataset['x']
        y = dataset['y']
        batch_size = len(x)
        nBatch = 1

        #dataset = break_into_batches(dataset, batch_size)
        
        loss_history = self.loss_history
        w_history    = self.w_history


        #data to batches

        self.optimizer.set_parameters({'learning_rate': learning_rate})


        for i in range(epochs):

            #for iB in range(0, nbatches):
            self.model.update_weights(w)

            loss_batch = 0.0
            dl_dw_batch = np.zeros(nVars)

            for iD in range(batch_size):

                x_i = x[iD]
                y_i = y[iD]

                self.model.zero_grad()

                y_pred        = self.model.forward(x_i)
                loss, dloss   = self.lossfun.eval(y_i, y_pred)
                self.model.backward(dloss)

                dl_dw = self.model.get_gradient()


                dl_dw_batch  += dl_dw
                loss_batch   += loss


            dl_dw_batch = dl_dw_batch/batch_size
            loss_batch  = loss_batch/batch_size

            self.loss_history.append(loss_batch)
            self.w_history.append(w)

            w = self.optimizer.step(w, dl_dw_batch)

            print("i : ",i, "loss : ", loss_batch)

        #Final output

        #calculate loos on final solution
        for iD in range(batch_size):

            x_i = x[iD]
            y_i = y[iD]

            self.model.zero_grad()

            y_pred        = self.model.forward(x_i)
            loss, dloss   = self.lossfun.eval(y_i, y_pred)
            self.model.backward(dloss)

            dl_dw = self.model.get_gradient()

            dl_dw_batch += dl_dw
            loss_batch   += loss


        dl_dw_batch = dl_dw_batch/batch_size
        loss_batch  = loss_batch/batch_size


        self.w_history.append(w)
        self.loss_history.append(loss_batch)


        solution = {
            "w": w,
            "dl_dw": dl_dw_batch,
            "loss": loss_batch
        }

        return solution


    
    




    def visualize(self):

        #Loss function history
        plt.plot(self.optimizer.loss_history)
        plt.show()

        #dvs


    def export_model(self):
        print("Exporting the optimizer")



    def import_model(self):
        print("Importing the optimizer")
        #loading a starting point for the optmization
  


#----------------------------------------------------------------------


def main():

    opti = Optimizer()

    opti.set_method('gradient_descent')

    opti.set_parameters({'learning_rate': 0.01, 
                        'max_iter': 100, 
                        'tol': 1e-6, 
                        'batch_size': 10})


    opti.set_variables(nVars, values=None)


    opti.set_model(net, lossfun)

   



#----------------------------------------------------------------------

if __name__ == "__main__":
    main()


#----------------------------------------------------------------------
