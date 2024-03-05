# CS6375 - Machine Learning
## Assignment-1 Part 2

### Problem Statement:
In this part, you will code a neural network (NN) having at least one hidden layers, besides the input and output layers. You are required to pre-process the data and then run the processed data through your neural net. Below are the requirements and suggested steps of the program.

- The programming language for this assignment will be Python 3.x
- You cannot use any libraries for neural net creation. You are free to use any other libraries for data loading, pre-processing, splitting, model evaluation, plotting, etc.
- Your code should be in the form of a Python class with methods like pre-process, train, test within the class. I leave the other details up to you.
- As the first step, pre-process and clean your dataset. There should be a method that does this.
- Split the pre-processed dataset into training and testing parts. You are free to choose any reasonable value for the train/test ratio, but be sure to mention it in the README file.
- Code a neural net having at least one hidden layer. You are free to select the number of neurons in each layer. Each neuron in the hidden and output layers should have a bias connection.
- You are required to add an optimizer on top of the basic backpropagation
algorithm. This could be the one you selected in the previous assigment
or a new one. Some good resources for gradient descent optimizers are: 
  - [An overview of gradient descent optimization algorithms](https://arxiv.org/pdf/1609.04747.pdf)
  - [An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/)
- You are required to code three different activation functions:
  1. Sigmoid 
  2. Tanh
  3. ReLu
  The earlier part of this assignment may prove useful for this stage. The activation function should be a parameter in your code.
- Code a method for creating a neural net model from the training part of the dataset. Report the training accuracy.
- Apply the trained model on the test part of the dataset. Report the test accuracy.
- You have to tune model parameters like learning rate, activation functions, etc. Report your results in a tabular format, with a column indicating the parameters used, a column for training accuracy, and one for test accuracy.

### Dataset: 
- Source: 
- Description

### Execution Instructions:

### File Structure