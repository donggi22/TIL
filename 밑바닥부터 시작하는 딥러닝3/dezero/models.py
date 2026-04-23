import numpy as np
from dezero import Layer
import dezero.functions as F
import dezero.layers as L
from dezero import utils

class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)