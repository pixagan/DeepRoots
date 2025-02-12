##Created : Anil Variyar
#last Modified : Anil Variyar | 12-02-2025
#Copyright : Anil Variyar | 2024 - 
#Terms of Use : MIT License | Read Terms and Conditions in the License file 
#No warranty of any kind is provided. Use at your own risk.


#An outline for an optimizer
#Not yet functional

import numpy as np

class Optimizer:

    def __init__(self, algorithm='gradient_descent', parameters={}):
        self.model     = None
        self.params    = {}
        self.dvs       = None
        self.optimizer = None


    def set_method(self, algorithm):

        if algorithm == 'gradient_descent':
            self.optimizer = GradientDescent()

        else:
            self.optimizer = GradientDescent()


    def set_parameters(self, parameters):

        self.params = parameters


    def set_variables(self, nVars, values=None, init_type='random'):

        if(values == None):
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




    def set_model(self, model):

        self.model = model


    def run(self):

        self.optimizer.run(self.model, self.dvs, self.params)



#----------------------------------------------------------------------


def main():

    opti = Optimizer()

    opti.set_method('gradient_descent')

    opti.set_parameters({'learning_rate': 0.01, 
                        'max_iter': 100, 
                        'tol': 1e-6, 
                        'batch_size': 10})


    opti.set_variables(nVars, values=None)

   



#----------------------------------------------------------------------

if __name__ == "__main__":
    main()


#----------------------------------------------------------------------
