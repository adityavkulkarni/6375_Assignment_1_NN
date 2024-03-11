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
#### Dataset 1: Heart disease(Processed Cleveland Dataset)
- Source: [Cleveland Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/data)
- Description:

#### Dataset 2: Breast Cancer
- Source: [Breast Cancer Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)

#### Dataset 3: Bank Churn Dataset
- Source: [Bank Churn Modelling](https://www.kaggle.com/datasets/filippoo/deep-learning-az-ann/download?datasetVersionNumber=1)
- Description:
  A bank is investigating a very high rate of customer leaving the bank. Here is a 10.000 records dataset to investigate and predict which of the customers are more likely to leave the bank soon.
  The data has following features:
       
     |   | CreditScore | Age | Tenure | Balance | NumOfProducts | HasCrCard | IsActiveMember | EstimatedSalary | Exited | 
     |--|---|---|---|---|---|---|---|---|---|
     | count | 10000.000000 | 10000.000000 | 10000.000000 | 10000.000000 | 10000.000000 | 10000.00000 | 10000.000000 | 10000.000000 | 10000.000000 | 
     | mean | 650.528800 | 38.921800 | 5.012800 | 76485.889288 | 1.530200 | 0.70550 | 0.515100 | 100090.239881 | 0.203700 | 
     | std | 96.653299 | 10.487806 | 2.892174 | 62397.405202 | 0.581654 | 0.45584 | 0.499797 | 57510.492818 | 0.402769 | 
     | min | 350.000000 | 18.000000 | 0.000000 | 0.000000 | 1.000000 | 0.00000 | 0.000000 | 11.580000 | 0.000000 | 
     | 25% | 584.000000 | 32.000000 | 3.000000 | 0.000000 | 1.000000 | 0.00000 | 0.000000 | 51002.110000 | 0.000000 | 
     | 50% | 652.000000 | 37.000000 | 5.000000 | 97198.540000 | 1.000000 | 1.00000 | 1.000000 | 100193.915000 | 0.000000 | 
     | 75% | 718.000000 | 44.000000 | 7.000000 | 127644.240000 | 2.000000 | 1.00000 | 1.000000 | 149388.247500 | 0.000000 | 
     | max | 850.000000 | 92.000000 | 10.000000 | 250898.090000 | 4.000000 | 1.00000 | 1.000000 | 199992.480000 | 1.000000 |


### Execution Instructions:

### File Structure:
``` .
├── ReadME.md
├── bu.py.bu : backup file
├── data
│   ├── Churn_Modelling.csv : dataset
│   └── sample.csv : sample dataset
├── eda.ipynb : notebook for data exploration and misc tasks 
├── main.py : main file
├── neural_net.py : file containing ANN and related code
├── nn-enhanced.pdf : assignment description
├── out : directory for storing output
└── utils.py : additional utils required for ANN
```