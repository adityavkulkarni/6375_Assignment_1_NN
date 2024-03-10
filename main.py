import pandas as pd
from sklearn.model_selection import train_test_split

from neural_net import NeuralNet
from utils import print_d


TRAINING_RATIO = 0.80


def load_data(path, train_size=None):
    if train_size is None:
        train_size = TRAINING_RATIO
    debug = train_size > 0
    data = pd.read_csv(path)
    data.dropna()
    print_d(f"Data rows loaded: {len(data)}", debug=debug)

    # Splitting the data into training and test sets (adjust test_size as needed)
    X = data.drop(columns=['Exited'])  # Features
    y = data['Exited']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    return pd.concat([X_train, y_train], axis=1), pd.concat([X_test, y_test], axis=1)
    # shuffle_df = data.sample(frac=1)
    # train_size = int(train_size * len(data))
    # tr = shuffle_df[:train_size]
    # ts = shuffle_df[train_size:]


if __name__ == '__main__':
    train_df, test_df = load_data('data/Churn_Modelling.csv')
    nn = NeuralNet(activation_function="sigmoid", debug=True)
    nn.train(training_data=train_df, test_data=test_df,
             learning_rate=0.1, epochs=100)
    nn.test()

    nn1 = NeuralNet(activation_function="tanh", debug=True)
    nn1.train(training_data=train_df, test_data=test_df,
              learning_rate=0.1, epochs=100)
    nn1.test()

    nn4 = NeuralNet(activation_function="relu", debug=True)
    nn4.train(training_data=train_df, test_data=test_df,
              learning_rate=0.1, epochs=100)
    nn4.test()

    nn2 = NeuralNet(activation_function="sigmoid", debug=True)
    nn2.train(training_data=train_df, test_data=test_df,
              learning_rate=0.1, epochs=100, optimizer="momentum")
    nn2.test()

    nn3 = NeuralNet(activation_function="tanh", debug=True)
    nn3.train(training_data=train_df, test_data=test_df,
              learning_rate=0.1, epochs=100, optimizer="momentum")
    nn3.test()

    nn3 = NeuralNet(activation_function="relu", debug=True)
    nn3.train(training_data=train_df, test_data=test_df,
              learning_rate=0.1, epochs=100, optimizer="momentum")
    nn3.test()
