import numpy as np
import pandas as pd


def prep_data(data_frame):
    useless_columns = [
        "BROKERTITLE",
        "ADDRESS",
        "STATE",
        "MAIN_ADDRESS",
        "ADMINISTRATIVE_AREA_LEVEL_2",
        "LOCALITY",
        "STREET_NAME",
        "LONG_NAME",
        "FORMATTED_ADDRESS"
    ]
    columns_to_encode = ["TYPE", "SUBLOCALITY"]
    data_frame = data_frame.drop(useless_columns, axis=1)
    one_hot_encoded = pd.get_dummies(data_frame, columns=columns_to_encode)
    one_hot_encoded = one_hot_encoded.astype(int)
    return one_hot_encoded


def normalize(df):
    # Z-score normalization
    mean_vals = df.mean()
    std_dev_vals = df.std()
    df_normalized = (df - mean_vals) / std_dev_vals
    df_normalized['LATITUDE'] = df['LATITUDE']/100
    df_normalized['LONGITUDE'] = df['LONGITUDE']/100 * -1
    return df_normalized


def data_loader():
    train_data = pd.read_csv("train_data.csv")
    train_label = pd.read_csv("train_label.csv")
    test_data = pd.read_csv("test_data.csv")

    train_data = prep_data(train_data)
    test_data = prep_data(test_data)

    # Remove additional columns from the test DataFrame and reindex
    columns_to_drop_from_train = [
        col for col in train_data.columns if col not in test_data.columns
    ]
    train_data = train_data.drop(columns=columns_to_drop_from_train)
    columns_to_drop_from_test = [
        col for col in test_data.columns if col not in train_data.columns
    ]
    test_data = test_data.drop(columns=columns_to_drop_from_test)
    test_data.reindex(train_data.columns, axis=1)

    # normalize the training data and test data
    test_data = normalize(test_data)
    train_data = normalize(train_data)

    return train_data.T, train_label.T, test_data.T


def init_params(input_size, hidden_size_1, hidden_size_2, output_size):
    weights1 = np.random.rand(hidden_size_1, input_size) - 0.5
    biases1 = np.random.rand(hidden_size_1, 1) - 0.5
    weights2 = np.random.rand(hidden_size_2, hidden_size_1) - 0.5
    biases2 = np.random.rand(hidden_size_2, 1) - 0.5
    weights3 = np.random.rand(output_size, hidden_size_2) - 0.5
    biases3 = np.random.rand(output_size, 1) - 0.5
    return weights1, biases1, weights2, biases2, weights3, biases3


def ReLU(data):
    return np.maximum(data, 0)


def softmax(data):
    output = np.exp(data) / sum(np.exp(data))
    return output


def forward_prop(weights1, biases1, weights2, biases2, weights3, biases3, X):
    result1 = weights1.dot(X) + biases1
    output1 = ReLU(result1)
    result2 = weights2.dot(output1) + biases2
    output2 = ReLU(result2)
    result3 = weights3.dot(output2) + biases3
    output3 = softmax(result3)
    return result1, output1, result2, output2, result3, output3


def ReLU_deriv(dataframe):
    return dataframe > 0


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 51))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backward_prop(
    result1, op1, result2, op2, result3, op3, weights1, weights2, weights3, X, Y
):
    x_size = X.shape[1]
    one_hot_Y = one_hot(Y)
    dres3 = op3 - one_hot_Y
    dweights3 = 1 / x_size * dres3.dot(op2.T)
    dbiases3 = 1 / x_size * np.sum(dres3)
    dres2 = weights3.T.dot(dres3) * ReLU_deriv(result2)
    dweights2 = 1 / x_size * dres2.dot(op1.T)
    dbiases2 = 1 / x_size * np.sum(dres2)

    dres1 = weights2.T.dot(dres2) * ReLU_deriv(result1)
    dweights1 = 1 / x_size * dres1.dot(X.T)
    dbiases1 = 1 / x_size * np.sum(dres1)
    return dweights1, dbiases1, dweights2, dbiases2, dweights3, dbiases3


def update_params(
    weights1,
    biases1,
    weights2,
    biases2,
    weights3,
    biases3,
    dweights1,
    dbiases1,
    dweights2,
    dbiases2,
    dweights3,
    dbiases3,
    learning_rate,
):
    weights1 = weights1 - learning_rate * dweights1
    biases1 = biases1 - learning_rate * dbiases1
    weights2 = weights2 - learning_rate * dweights2
    biases2 = biases2 - learning_rate * dbiases2
    weights3 = weights3 - learning_rate * dweights3
    biases3 = biases3 - learning_rate * dbiases3
    return weights1, biases1, weights2, biases2, weights3, biases3


def get_predictions(data):
    return np.argmax(data, axis=0)


def get_accuracy(predictions, Y):
    y_transformed = np.array(Y)[0]
    return np.sum(predictions == y_transformed) / Y.size


def gradient_descent(X, Y, learning_rate, epochs):
    input_size = X.shape[0]
    hidden_size_1 = 64  # Number of neurons in the hidden layer
    hidden_size_2 = 64  # Number of neurons in the hidden layer
    output_size = 51  # Number of neurons in the output layer
    weights1, biases1, weights2, biases2, weights3, biases3 = init_params(
        input_size, hidden_size_1, hidden_size_2, output_size
    )
    for i in range(epochs):
        res1, op1, res2, op2, res3, op3 = forward_prop(
            weights1, biases1, weights2, biases2, weights3, biases3, X
        )
        dweights1, dbiases1, dweights2, dbiases2, dweights3, dbiases3 = backward_prop(
            res1, op1, res2, op2, res3, op3, weights1, weights2, weights3, X, Y
        )
        weights1, biases1, weights2, biases2, weights3, biases3 = update_params(
            weights1,
            biases1,
            weights2,
            biases2,
            weights3,
            biases3,
            dweights1,
            dbiases1,
            dweights2,
            dbiases2,
            dweights3,
            dbiases3,
            learning_rate,
        )
    return weights1, biases1, weights2, biases2, weights3, biases3


def make_predictions(X, weights1, biases1, weights2, biases2, weights3, biases3):
    _, _, _, _, _, result = forward_prop(
        weights1, biases1, weights2, biases2, weights3, biases3, X
    )
    predictions = get_predictions(result)
    return predictions


# USE_DATASET_SPLIT 2
### Data processing ###
X_train, Y_train, X_test = data_loader()
weights1, biases1, weights2, biases2, weights3, biases3 = gradient_descent(
    X_train, Y_train, 0.21, 1000
)
predictions = make_predictions(
    X_test, weights1, biases1, weights2, biases2, weights3, biases3
)

with open("output.csv", "w") as f:
    f.write("BEDS\n")
    for value in predictions:
        f.write("%s\n" % value)
