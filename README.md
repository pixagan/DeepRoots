# DeepRoots
Built to help students understand the basics of neurons and neural networks.

Allous learners to assemble simple networks.
Also setup to allow learners and researchers to visualize the networks and the optimization process.



## Getting started

### Download the repository from github

```bash
git clone https://github.com/pixagan/deeproots.git
```

### Install the dependencies

```bash
pip install -r requirements.txt
```

### Install the package locally using pip

```bash
pip install -e .
```


## Creating a Multi Layer Perceptron

### A Simple Dataset

```python
X_train = np.array([[1,2,3], [4,5,6]])
y_train = np.array([[1,0], [0,1]])
```


### Create the tanh Layers
```
L1 = LayerScalar(3, 4, 'tanh')
L2 = LayerScalar(4, 2, 'tanh')
```


### Creating the Network

```
layer_list = [L1, L2]
net = NNScalar(layer_list)
```

### Setting up a loss function

```python
lossfun = SoftmaxCrossEntropy()
```

### Picking up a data 

```python
x_act = X_train[0,:]
y_act = y_train[0,:]
```

### Forward pass through the network

```python
y_pred      = net.forward(x_act)
```

### Evaluating the loss function

```python
loss, dL_dy = lossfun.eval(y_act, y_pred)
```

### Backward pass through the network

```python
net.backward(dL_dy)
```

### Getting the gradients

```python
grad = net.get_gradient()
```

### Getting the weights

```python
weights = net.get_weights()
print("weights ", weights)
```









