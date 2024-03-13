import argparse

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from neural_net import NeuralNet, LabelEncoderExt

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
    parser.add_argument('--activation', type=str, choices=['sigmoid', 'tanh', 'relu', 'all'],
                        help='Activation function to use: sigmoid, tanh, relu, or all', default="all")
    parser.add_argument('--optimizer', type=str, choices=['none', 'momentum', 'all'],
                        help='Optimizer to use: none, momentum, or all', default="all")

    args = parser.parse_args()

    compare = []
    print("Heart disease dataset")
    d = {"Dataset": "Heart disease"}
    train_df, val_df = load_data('data/heart.csv',
                                 target_col='target')
    nn_s = NeuralNet(activation_function="sigmoid", hidden_layer_size=[9, 6, 4])
    nn_s.train(training_data=train_df, test_data=val_df,
               learning_rate=0.1, epochs=100, plot_suffix="heart")
    d["Sigmoid"] = nn_s.test()[1]
    nn_s_m = NeuralNet(activation_function="sigmoid", hidden_layer_size=[9, 6, 4])
    nn_s_m.train(training_data=train_df, test_data=val_df,
                 learning_rate=0.005, epochs=1000, optimizer="momentum", plot_suffix="heart")
    d["Sigmoid Momentum"] = nn_s_m.test()[1]

    nn_t = NeuralNet(activation_function="tanh", hidden_layer_size=[9, 6, 4])
    nn_t.train(training_data=train_df, test_data=val_df,
               learning_rate=0.05, epochs=100, plot_suffix="heart")
    d["Tanh"] = nn_t.test()[1]
    nn_t_m = NeuralNet(activation_function="tanh", hidden_layer_size=[9, 6, 4])
    nn_t_m.train(training_data=train_df, test_data=val_df,
                 learning_rate=0.001, epochs=1000, optimizer="momentum", plot_suffix="heart")
    d["Tanh Momentum"] = nn_t_m.test()[1]

    nn_r = NeuralNet(activation_function="relu", hidden_layer_size=[7, 3])
    nn_r.train(training_data=train_df, test_data=val_df,
               learning_rate=0.01, epochs=100, plot_suffix="heart")
    d["ReLU"] = nn_r.test()[1]
    nn_r_m = NeuralNet(activation_function="relu", hidden_layer_size=[9, 6, 4])
    nn_r_m.train(training_data=train_df, test_data=val_df,
                 learning_rate=0.001, epochs=1000, optimizer="momentum", plot_suffix="heart")
    d["ReLU Momentum"] = nn_r_m.test()[1]
    compare.append(d)

    print("Breast Cancer dataset")
    train_df, val_df = load_data('data/breast-cancer.csv', target_col='diagnosis')
    train_df.drop('id', axis=1, inplace=True)
    val_df.drop('id', axis=1, inplace=True)
    train_df['diagnosis'] = (train_df['diagnosis'] == 'M').astype(int)
    val_df['diagnosis'] = (val_df['diagnosis'] == 'M').astype(int)
    perf = []
    d = {"Dataset": "Breast cancer"}
    nn_s = NeuralNet(activation_function="sigmoid", hidden_layer_size=[20, 13, 9])
    nn_s.train(training_data=train_df, test_data=val_df,
               learning_rate=0.1, epochs=100, plot_suffix="cancer")
    d["Sigmoid"] = nn_s.test()[1]
    nn_s_m = NeuralNet(activation_function="sigmoid", hidden_layer_size=[20, 13, 9])
    nn_s_m.train(training_data=train_df, test_data=val_df,
                 learning_rate=0.008, epochs=1000, optimizer="momentum", plot_suffix="cancer")
    d["Sigmoid Momentum"] = nn_s_m.test()[1]

    nn_t = NeuralNet(activation_function="tanh", hidden_layer_size=[20, 13, 9])
    nn_t.train(training_data=train_df, test_data=val_df,
               learning_rate=0.1, epochs=100, plot_suffix="cancer")
    d["Tanh"] = nn_t.test()[1]
    nn_t_m = NeuralNet(activation_function="tanh", hidden_layer_size=[20, 13, 9])
    nn_t_m.train(training_data=train_df, test_data=val_df,
                 learning_rate=0.0005, epochs=1000, optimizer="momentum", plot_suffix="cancer")
    d["Tanh Momentum"] = nn_t_m.test()[1]

    nn_r = NeuralNet(activation_function="relu", hidden_layer_size=[21, 7, 3])
    nn_r.train(training_data=train_df, test_data=val_df,
               learning_rate=0.03, epochs=100, plot_suffix="cancer")
    d["ReLU"] = nn_r.test()[1]
    nn_r_m = NeuralNet(activation_function="relu", hidden_layer_size=[21, 7, 3])
    nn_r_m.train(training_data=train_df, test_data=val_df,
                 learning_rate=0.009, epochs=1000, optimizer="momentum", plot_suffix="cancer")
    d["ReLU Momentum"] = nn_r_m.test()[1]
    compare.append(d)

    compare = pd.DataFrame(compare)
    compare.set_index('Dataset', inplace=True)
    ax = compare.plot(kind='bar', figsize=(12, 7), rot=0)

    # Add title and labels
    plt.title('Performance of Different Models on Various Datasets')
    plt.xlabel('Model')
    plt.ylabel('Performance')

    # Add legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(title='Dataset', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f"./out/compare.png")
    plt.close()
    compare.to_csv("./out/compare.csv", index=False)
