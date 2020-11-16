# Number Reader Leaky ReLu

Leaky ReLu neural network for hand drawn number recognition that I did for fun.
I didn't had any experience with neural networks before (just experience with numerical optimization)
 so I am very happy with the results.
 
Each input neuron reads a pixel of a image (so, 784 input neurons for the 28x28 MNIST data set)
 and each output neuron represents a number (so, 10 output neurons), all hidden layers have the same number of neurons
 (4 hidden layers with 16 neurons each for the test neural network on NumberReaderData.yaml). It uses a Leaky ReLu
 (Leaky rectified linear unit) for the activation function, evolutive rate of 0.1 and an stochastic gradient descent
 with momentum of 0.75 as the optimization algorithm and a L2 regularization treshold of 1. Batch training with 100 images per batch.
 
It can reach around 93% of precision in the MNIST data set with 20 epochs of training and it only uses numpy and pyyalm as external libraries for matrix operations and data persistence respectively.

###Usage:

`python Main.py test` to run it against the MNIST test data set.

`python Main.py retrain` to retrain it with 20 epochs of the MNIST training data set and save the new neural network.

`Python Main.py train #n` where #n is an integer to further train a saved neural network with n epochs of the MNIST training data set.

You can change the input neurons, number of hidden layers and neurons by hidden layer in Main.py.\
Evolutive rate and momentum are defined in NumberReader.py.

NumberReaderData.yaml has a trained neural network (4 hidden layers, 16 neurons by hidden layer, 20 epochs) so you can test it without needing to train it first.