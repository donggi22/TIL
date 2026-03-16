import numpy as np
from dezero.core_simple import Variable
from dezero.utils import plot_dot_graph

x = Variable(np.array(3.0))
y = Variable(np.array(0.0))

while True:
    y = y + x
    if y.data > 100:
        break

y.backward()
print(x.grad)

x.name = 'x'
y.name = 'y'
plot_dot_graph(y, verbose=False, to_file='반복문.png')