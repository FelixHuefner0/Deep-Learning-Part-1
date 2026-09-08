"""
Part1 - Ref conent: "Artificial neurons" of script 

"""

from math import exp
import numpy as np
import matplotlib.pyplot as plt


class LogisticNeuron1D:
    """
    Representing one-dimensional logistic neurons.
    """
    def __init__ (self, weight:float=1, bias:float=0):
        self.weight = weight
        self.bias = bias
    def eval (self, x:float):
        """
        Evaluate the neuron at an input point.
        """
        #affine part:
            #x -> w*x + b
        #activation:
            #x -> 1 / exp(-x)
        return 1/(1+exp(-self.weight*x-self.bias))
    def plot (self, x_min=-10, x_max=10, num_points=1000):
        """
        Plot the graph of the neuron.
        """
        plt.title("Weight="+str(self.weight)+", Bias="+str(self.bias))
        grid = np.linspace(start=x_min, stop=x_max, num=num_points)
        plt.plot(grid, [self.eval(x) for x in grid], color="black")
        plt.show()

#The 'standard' logistic neuron: weight=1, bias=0
LogisticNeuron1D(weight=1, bias=0).plot(-10,10)

#The 'standard' graph can be translated by adjusting the bias
LogisticNeuron1D(weight=1, bias=-5).plot(-10,10)

#The 'standard' graph can be made to increase faster by adjusting the weight
LogisticNeuron1D(weight=10, bias=0).plot(-10,10)

#Both at the same time
LogisticNeuron1D(weight=10, bias=-5*10).plot(-10,10)

#General method to obtain an arbitrarily precise approximation to a hard threshold function
#set the desired threshold
threshold = -3
#alpha controls how close the neuron is to a step function
#even relatively small values (e.g. alpha=100)
#can already be to large for the functions in python's the math libary
alpha = 5
#calculate weight and bias
weight = alpha
bias = -alpha * threshold
#plot the neuron
LogisticNeuron1D(weight=weight, bias=bias).plot(-10,10)
