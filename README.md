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

### Create the tanh Layers
L1 = LayerScalar(3, 4, 'tanh')
L2 = LayerScalar(4, 2, 'tanh')


### Creating the Network
layer_list = [L1, L2]
net = NNScalar(layer_list)


### Load the Data using the DataLoader class

### Set the weights of the network


### Run the Data through the network


## Train the Network

### Data into training and testing sets

### Get the one-hot encoding

### Add a loss function


### Initialize the optimizer


### Run the optimizer











