import numpy as np
import pandas as pd

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

    shuffle_df = data.sample(frac=1)
    train_size = int(train_size * len(data))
    tr = shuffle_df[:train_size]
    ts = shuffle_df[train_size:]
    print_d(f"Training data size: {len(tr)}", debug=debug)
    print_d(f"Testing data size: {len(ts)}", debug=debug)
    return tr, ts


if __name__ == '__main__':
    train_df, test_df = load_data('data/Churn_Modelling.csv')
    _, sample_df = load_data('data/sample.csv', train_size=0)
    nn = NeuralNet(debug=True)
    nn.train(training_data=train_df, test_data=test_df, learning_rate=0.1, epochs=100)
    for _, row in sample_df.iterrows():
        nn.predict(row)
        print_d(f"Sample {_}: {nn.predict(row)}", debug=True)