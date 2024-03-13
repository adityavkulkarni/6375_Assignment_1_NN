import argparse

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from neural_net import NeuralNet

plt.style.use('tableau-colorblind10')
TRAINING_RATIO = 0.75


def load_data(path, target_col="", train_size=None):
    data = pd.read_csv(path)
    data.dropna()

    # Splitting the data into training and test sets (adjust test_size as needed)
    X = data.drop(columns=[target_col])  # Features
    y = data[target_col]  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size,
                                                        random_state=33, shuffle=True, stratify=y)

    return pd.concat([X_train, y_train], axis=1), pd.concat([X_test, y_test], axis=1)
    # shuffle_df = data.sample(frac=1)
    # train_size = int(train_size * len(data))
    # tr = shuffle_df[:train_size]
    # ts = shuffle_df[train_size:]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train a neural network.')
    parser.add_argument('--dataset', type=str, choices=['cancer', 'heart', 'all'],
                        help='Dataset to use: bank, cancer, heart, or all', default="all")
    parser.add_argument('--activation', type=str, choices=['sigmoid', 'tanh', 'relu'],
                        help='Activation function to use: sigmoid, tanh, relu', default="sigmoid")
    parser.add_argument('--optimizer', type=str, choices=['none', 'momentum', 'adagrad'],
                        help='Optimizer to use: none, momentum, adagrad or all', default="none")
    parser.add_argument('--gradient', type=str, choices=['stochastic', 'batch', 'minibatch'],
                        help='Gradient descent type: batch, minibatch, stochastic',
                        default="batch")
    parser.add_argument('--learning-rate', type=float,
                        help='Learning rate for model', default=0.01)
    parser.add_argument('--epochs', type=int,
                        help='Epochs for model', default=0.01)
    args = parser.parse_args()

    res = []
    if args.dataset == 'all' or args.dataset == 'heart':
        print("Heart disease dataset")
        train_df, val_df = load_data('data/heart.csv',
                                     target_col='target')
        nn = NeuralNet(activation_function=args.activation, hidden_layer_size=[9, 6, 4])
        a, l, t = nn.train(training_data=train_df, test_data=val_df, gradient=args.gradient,
                           learning_rate=args.learning_rate, epochs=args.epochs, optimizer=args.optimizer,
                           plot_suffix="heart")
        res.append([
            args.activation, args.gradient, args.optimizer,
            args.learning_rate, args.epochs, a, l, t
        ])
    res_df = pd.read_csv('out/heart/heart_results.csv')
    new = pd.DataFrame(columns=res_df.columns, data=res)
    df = pd.concat([res_df, new], axis=0)
    df.to_csv('out/heart/heart_results.csv', index=False)

    if args.dataset == 'all' or args.dataset == 'cancer':
        print("Breast Cancer dataset")
        train_df, val_df = load_data('data/breast-cancer.csv', target_col='diagnosis')
        train_df.drop('id', axis=1, inplace=True)
        val_df.drop('id', axis=1, inplace=True)
        train_df['diagnosis'] = (train_df['diagnosis'] == 'M').astype(int)
        val_df['diagnosis'] = (val_df['diagnosis'] == 'M').astype(int)
        nn_s = NeuralNet(activation_function="sigmoid", hidden_layer_size=[20, 13, 9])
        a, l, t = nn_s.train(training_data=train_df, test_data=val_df, gradient=args.gradient,
                             learning_rate=args.learning_rate, epochs=args.epochs, optimizer=args.optimizer,
                             plot_suffix="cancer")
        res.append([
            args.activation, args.gradient, args.optimizer,
            args.learning_rate, args.epochs, a, l, t
        ])
    res_df = pd.read_csv('out/cancer/cancer_results.csv')
    new = pd.DataFrame(columns=res_df.columns, data=res)
    df = pd.concat([res_df, new], axis=0)
    df.to_csv('out/cancer/cancer_results.csv', index=False)
