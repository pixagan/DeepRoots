##Created : Anil Variyar
#last Modified : Anil Variyar | 12-02-2025
#Copyright : Anil Variyar | 2024 - 
#Terms of Use : MIT License | Read Terms and Conditions in the License file 
#No warranty of any kind is provided. Use at your own risk.

import numpy as np

class GradientDescent():

    def __init__(self):
        super().__init__()
        print("Initializing Gradient Descent")
        self.model = None
        self.his   = None


    def set_model(self, model):
        self.model = model


    def set_parameters(self, parameters):
        self.params = parameters


    def step(self, w, dw):
    
        #update the design variables
        w = w - self.params['learning_rate']*dw

        return w





    def run(self, iters):

        for i in range(iters):
            self.step(i)



    def run_old(self, iters):

        #Try out forward and backward iter

        #padding inputs with a 1

        iters    = np.arange(0,nEpochs)
        w_his    = np.zeros([nEpochs, len(w)])
        loss_his = np.zeros(nEpochs)

        dwT = np.zeros(len(w))

        net.update_weights(w)

        for ep in range(nEpochs):
            tloss = 0.0
            dwT = np.zeros(len(w))

            for i in range(0, len(X_train)):
            
                x_act = X_train[i]
                y_act = y_train[i]

                self.model()


                #write function calls
                y_hat         = net.forward(x_act)
                
                loss, dloss   = lossfun.eval(y_act, y_hat)
                
                net.backward(dloss)
                dL_dw         = net.get_gradient()
                #dL_dw         = dL_dw
                
            
                dwT   += dL_dw
                tloss = tloss + loss


            w_his[ep,:]  = w[:] 
            loss_his[ep] = tloss/n

            print("Loss ", ep, i, tloss/n)
                    
            w = w - d*dwT/n
            net.update_weights(w)


            

        #    if(ep % 10 == 0):
        #        print("Loss ", ep, tloss/n)




#----------------------------------------------------------------------


def main():

    opti = GradientDescent()
   


#----------------------------------------------------------------------

if __name__ == "__main__":
    main()


#----------------------------------------------------------------------

        

 