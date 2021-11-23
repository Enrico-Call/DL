import vugrad as vg
import numpy as np

# Create two Tensor nodes with
a = vg.TensorNode(np.random.randn(2, 2))
b = vg.TensorNode(np.random.randn(2, 2))

# Create another Tensor node with a sum operation

c = a + b

# Check the properties of c

print(c.value)
print(c.source)
print(c.source.inputs[0].value)
print(a.grad)