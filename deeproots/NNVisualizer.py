##Created : Anil Variyar
#last Modified : Anil Variyar | 12-02-2025
#Copyright : Anil Variyar | 2024 - 
#Terms of Use : MIT License | Read Terms and Conditions in the License file 
#No warranty of any kind is provided. Use at your own risk.

#Outline for a Visualization framework for Neural Networks

import streamlit as st
import numpy as np

class NNVisualizer:

    def __init__(self, modelType, model):
        self.modelType = modelType
        self.model = model

    def convert_to_native(self):
        pass

    def visualize(self):

        #no of layers

        #pick the layer - select button 

        #pick the neuron - select button 

        pass





class OptimizerVisualizer:

    def __init__(self, optimizerType, optimizer):
        self.optimizerType = optimizerType
        self.optimizer = optimizer

    def visualize(self):

        #visualize the loss curve

        #visualize the weights

        #visualize the gradients
        
        
        pass


#----------------------------------------------------------------------


def main():

    nnviz = NNVisualizer("Linear", LinearModel())
    nnviz.visualize()


#----------------------------------------------------------------------

if __name__ == "__main__":
    main()


#----------------------------------------------------------------------

     