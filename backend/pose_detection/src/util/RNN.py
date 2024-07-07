import keras


class RNN:

    def __init__(self, verbose=False):
        self.input = keras.layers.Input(shape=(13, 2))
        self.lstm1 = keras.layers.LSTM(128)
        self.dense1 = keras.layers.Dense(128, input_shape=(13, 2))
        self.dense2 = keras.layers.Dense(26)
        self.output = keras.layers.Reshape(target_shape=(13, 2))

        self.model = keras.Sequential()
        self.model.add(self.input)
        self.model.add(self.lstm1)
        self.model.add(self.dense1)
        self.model.add(self.dense2)
        self.model.add(self.output)

        if verbose:
            print(self.model.summary())

    def fit(self, data):
        print(data)
        self.model.fit(data)

    def predict(self, data):
        print(data)
        return self.model.predict(data)
