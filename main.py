import argparse

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from neural_net import NeuralNet, LabelEncoderExt
from utils import print_d

plt.style.use('tableau-colorblind10')
TRAINING_RATIO = 0.75
HEART = CANCER = BANK = False
SIGMOID = TANH = RELU = False
NONE = MOMENTUM = False


def set_var(arguments):
    global HEART, CANCER, BANK, SIGMOID, TANH, RELU, NONE, MOMENTUM
    if arguments.dataset == 'all':
        HEART = CANCER = BANK = True
    elif arguments.dataset == 'heart':
        HEART = True
    elif arguments.dataset == 'cancer':
        CANCER = True
    elif arguments.dataset == 'bank':
        BANK = True
    if arguments.activation == 'all':
        SIGMOID = TANH = RELU = True
    elif arguments.dataset == 'sigmoid':
        SIGMOID = True
    elif arguments.dataset == 'tanh':
        TANH = True
    elif arguments.dataset == 'relu':
        RELU = True
    if arguments.optimizer == 'all':
        NONE = MOMENTUM = True
    elif arguments.dataset == 'none':
        NONE = True
    elif arguments.dataset == 'momentum':
        MOMENTUM = True


def load_data(path, target_col="", train_size=None):
    if train_size is None:
        train_size = TRAINING_RATIO
    debug = train_size > 0
    data = pd.read_csv(path)
    data.dropna()
    print_d(f"Data rows loaded: {len(data)}", debug=debug)

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
    parser.add_argument('--dataset', type=str, choices=['bank', 'cancer', 'heart', 'all'],
                        help='Dataset to use: bank, cancer, heart, or all', default="all")
    parser.add_argument('--activation', type=str, choices=['sigmoid', 'tanh', 'relu', 'all'],
                        help='Activation function to use: sigmoid, tanh, relu, or all', default="all")
    parser.add_argument('--optimizer', type=str, choices=['none', 'momentum', 'all'],
                        help='Optimizer to use: none, momentum, or all', default="all")

    args = parser.parse_args()
    set_var(args)

    compare = []
    if HEART:
        print("Heart disease dataset")
        d = {"Dataset": "Heart disease"}
        train_df, val_df = load_data('data/heart.csv',
                                     target_col='target')
        if SIGMOID:
            nn_s = NeuralNet(activation_function="sigmoid")
            nn_s.train(training_data=train_df, test_data=val_df,
                       learning_rate=0.05, epochs=100, plot_suffix="heart")
            d["Sigmoid"] = nn_s.test()[1]
            if MOMENTUM:
                nn_s_m = NeuralNet(activation_function="sigmoid")
                nn_s_m.train(training_data=train_df, test_data=val_df,
                             learning_rate=0.01, epochs=100, optimizer="momentum", plot_suffix="heart")
                d["Sigmoid Momentum"] = nn_s_m.test()[1]
        if TANH:
            nn_t = NeuralNet(activation_function="tanh")
            nn_t.train(training_data=train_df, test_data=val_df,
                       learning_rate=0.05, epochs=100, plot_suffix="heart")
            d["Tanh"] = nn_t.test()[1]
            if MOMENTUM:
                nn_t_m = NeuralNet(activation_function="tanh")
                nn_t_m.train(training_data=train_df, test_data=val_df,
                             learning_rate=0.01, epochs=100, optimizer="momentum", plot_suffix="heart")
                d["Tanh Momentum"] = nn_t_m.test()[1]
        compare.append(d)

    if CANCER:
        print("Breast Cancer dataset")
        train_df, val_df = load_data('data/breast-cancer.csv', target_col='diagnosis')
        train_df.drop('id', axis=1, inplace=True)
        val_df.drop('id', axis=1, inplace=True)
        train_df['diagnosis'] = (train_df['diagnosis'] == 'M').astype(int)
        val_df['diagnosis'] = (val_df['diagnosis'] == 'M').astype(int)
        perf = []
        d = {"Dataset": "Breast cancer"}
        if SIGMOID:
            nn_s = NeuralNet(activation_function="sigmoid")
            nn_s.train(training_data=train_df, test_data=val_df,
                       learning_rate=0.05, epochs=100, plot_suffix="cancer")
            d["Sigmoid"] = nn_s.test()[1]
            if MOMENTUM:
                nn_s_m = NeuralNet(activation_function="sigmoid")
                nn_s_m.train(training_data=train_df, test_data=val_df,
                             learning_rate=0.01, epochs=100, optimizer="momentum", plot_suffix="cancer")
                d["Sigmoid Momentum"] = nn_s_m.test()[1]
        if TANH:
            nn_t = NeuralNet(activation_function="tanh")
            nn_t.train(training_data=train_df, test_data=val_df,
                       learning_rate=0.05, epochs=100, plot_suffix="cancer")
            d["Tanh"] = nn_t.test()[1]
            if MOMENTUM:
                nn_t_m = NeuralNet(activation_function="tanh")
                nn_t_m.train(training_data=train_df, test_data=val_df,
                             learning_rate=0.01, epochs=100, optimizer="momentum", plot_suffix="cancer")
                d["Tanh Momentum"] = nn_t_m.test()[1]
        compare.append(d)

    if BANK:
        print("Bank churn dataset")
        train_df, val_df = load_data('data/Churn_Modelling.csv', target_col='Exited')
        train_df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True, errors="ignore")
        val_df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True, errors="ignore")
        encoder_geo = LabelEncoderExt()
        encoder_gender = LabelEncoderExt()
        encoder_geo.fit(train_df["Geography"])
        encoder_gender.fit(train_df["Gender"])
        train_df["Gender"] = encoder_gender.transform(train_df["Gender"])
        train_df["Geography"] = encoder_geo.transform(train_df["Geography"])
        val_df["Gender"] = encoder_gender.transform(val_df["Gender"])
        val_df["Geography"] = encoder_geo.transform(val_df["Geography"])

        d = {"Dataset": "Bank churn"}
        if SIGMOID:
            nn_s = NeuralNet(activation_function="sigmoid")
            nn_s.train(training_data=train_df, test_data=val_df,
                       learning_rate=0.05, epochs=50, plot_suffix="bank")
            d["Sigmoid"] = nn_s.test()[1]
            if MOMENTUM:
                nn_s_m = NeuralNet(activation_function="sigmoid")
                nn_s_m.train(training_data=train_df, test_data=val_df,
                             learning_rate=0.01, epochs=50, optimizer="momentum", plot_suffix="bank")
                d["Sigmoid Momentum"] = nn_s_m.test()[1]
        if TANH:
            nn_t = NeuralNet(activation_function="tanh")
            nn_t.train(training_data=train_df, test_data=val_df,
                       learning_rate=0.05, epochs=50, plot_suffix="bank")
            d["Tanh"] = nn_t.test()[1]
            if MOMENTUM:
                nn_t_m = NeuralNet(activation_function="tanh")
                nn_t_m.train(training_data=train_df, test_data=val_df,
                             learning_rate=0.01, epochs=50, optimizer="momentum", plot_suffix="bank")
                d["Tanh Momentum"] = nn_t_m.test()[1]
        compare.append(d)
    compare = pd.DataFrame(compare)
    compare.set_index('Dataset', inplace=True)
    ax = compare.plot(kind='bar', figsize=(16, 9), rot=0)

    # Add title and labels
    plt.title('Performance of Different Models on Various Datasets')
    plt.xlabel('Model')
    plt.ylabel('Performance')

    # Add legend
    plt.legend(title='Dataset')
    plt.savefig(f"./out/compare.png")
    plt.close()
    compare.to_csv("./out/compare.csv", index=False)
