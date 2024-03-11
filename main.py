import pandas as pd
from sklearn.model_selection import train_test_split

from neural_net import NeuralNet, LabelEncoderExt
from utils import print_d


TRAINING_RATIO = 0.75


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
    print("Heart disease dataset")
    train_df, val_df = load_data('data/heart.csv',
                                 target_col='target')

    nn1 = NeuralNet(activation_function="sigmoid")
    nn1.train(training_data=train_df, test_data=val_df,
              learning_rate=0.05, epochs=100, plot_suffix="heart")

    nn2 = NeuralNet(activation_function="sigmoid")
    nn2.train(training_data=train_df, test_data=val_df,
              learning_rate=0.01, epochs=100, optimizer="momentum", plot_suffix="heart")

    print("Breast Cancer dataset")
    train_df, val_df = load_data('data/breast-cancer.csv', target_col='diagnosis')
    train_df.drop('id', axis=1, inplace=True)
    val_df.drop('id', axis=1, inplace=True)
    train_df['diagnosis'] = (train_df['diagnosis'] == 'M').astype(int)
    val_df['diagnosis'] = (val_df['diagnosis'] == 'M').astype(int)

    nn3 = NeuralNet(activation_function="sigmoid", hidden_layer_size=[20, 13, 9])
    nn3.train(training_data=train_df, test_data=val_df,
              learning_rate=0.05, epochs=200, plot_suffix="cancer")

    nn4 = NeuralNet(activation_function="sigmoid", hidden_layer_size=[20, 13, 9])
    nn4.train(training_data=train_df, test_data=val_df,
              learning_rate=0.05, epochs=100, optimizer="momentum", plot_suffix="cancer")

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

    nn5 = NeuralNet(activation_function="sigmoid", hidden_layer_size=[10, 7, 5])
    nn5.train(training_data=train_df, test_data=val_df,
              learning_rate=0.01, epochs=100, plot_suffix="bank")

    nn6 = NeuralNet(activation_function="sigmoid", hidden_layer_size=[10, 7, 5])
    nn6.train(training_data=train_df, test_data=val_df,
              learning_rate=0.01, epochs=100, optimizer="momentum", plot_suffix="bank")
