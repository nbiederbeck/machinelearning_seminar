from keras.datasets import mnist
from keras.utils import np_utils 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class dataset:
    def __init__(self):
        ''' load dataset and reshape it from shape 
        (xxx, 28, 28) -> (xxx, 1, 28, 28) for keras, 
        convert values from int to float and encode
        Y-values for predictions.
        '''
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        img_rows = X_train.shape[1]
        img_cols = X_train.shape[2]
        N_train = X_train.shape[0]
        N_test = X_test.shape[0]
        self.X_train = X_train.reshape(N_train, img_rows*img_cols)
        self.X_train = self.X_train.astype('float32')
        self.X_test= X_test.reshape(N_test, img_rows*img_cols)
        self.X_test = self.X_test.astype('float32')
        self.Y_train = np_utils.to_categorical(y_train, 10)
        self.Y_test = np_utils.to_categorical(y_test, 10)

        scaler = MinMaxScaler(feature_range=(0, 1))
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        self.X_train = self.X_train.reshape(N_train, img_rows, img_cols, 1)
        self.X_test = self.X_test.reshape(N_test, img_rows, img_cols, 1)


    def get_train_val_data(self, test_size = 0.3):
        ''' Returns Training data.
        Params:
            test_size: float (0,1)
                size of the testset
        Returns:
            X_train, X_test: np.array
                Inputdata
            Y_train, Y_test: np.array
                Label
        '''
        X_train, X_val, Y_train, Y_test = train_test_split(
                self.X_train, self.Y_train, 
                test_size = test_size, 
                stratify= self.Y_train,
                random_state = 42
                )
        return X_train, Y_train, X_val, Y_test
    
    def get_test_data(self):
        '''Return test data.
        Return:
            X_test: np.array
                Inputdata
            Y_test: np.array
                Label
        '''
        return self.X_test, self.Y_test

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
# %%%%%%%%%%%%%               Network             %%%%%%%%%%%%%%%%%%%%%%%%%%% 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

from keras.models import Sequential
from keras.layers import GaussianNoise
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import model_from_yaml
import pandas as pd

class network:
    def __init__(self, batch_size=512, epochs=100):
        self.model = Sequential()
        self.batch_size = batch_size
        self.epochs = epochs
        self.history = None

    def add_layer(self, layer):
        self.model.add(layer)

    def compile(self, loss='categorical_crossentropy', 
            optimizer='adam', metrics=['accuracy']):
        self.model.compile(loss=loss,
              optimizer=optimizer,
              metrics=metrics)
        self.model.summary()


    def fit(self, X_train, Y_train, X_val, Y_val):
        callback = self.model.fit(
                X_train, Y_train, validation_data = (X_val, Y_val), 
                batch_size = self.batch_size, epochs= self.epochs, 
                verbose = 1
                )
        self.history = pd.DataFrame(callback.history)
    
    def evaluate(self, X_test, Y_test):
        score = self.model.fit(
                X_test, Y_test, batch_size = self.batch_size                )
        return score.history

    def save(self, file_path):
        model_yaml = self.model.to_yaml()
        with open(file_path + "model.yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
        self.model.save_weights(file_path + "model.h5")
        self.history.to_pickle(file_path + "history.pkl")
        print("(Neuronal Network) Saved model to disk")


    def load(self, file_path, compile=True):
        yaml_file = open(file_path + 'model.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        self.model = model_from_yaml(loaded_model_yaml)
        self.model.load_weights(file_path + "model.h5")
        print("(Neuronal Network) Loaded model from disk")
        if compile:
            self.compile()
    
def main():
    data = dataset()

    net = network(epochs=20)
    net.add_layer(Conv2D(6, kernel_size=(5, 5), padding='valid',
        activation='relu', input_shape=(28,28,1)))
    net.add_layer(MaxPooling2D((4,4)))
    net.add_layer(Conv2D(12, kernel_size=(4, 4), padding='valid',
        activation='relu', input_shape=(28,28,1)))
    net.add_layer(Flatten())
    net.add_layer(Dropout(0.2))
    net.add_layer(Dense(32, activation='relu'))
    net.add_layer(GaussianNoise(0.2))
    net.add_layer(Dropout(0.2))
    net.add_layer(Dense(10, activation='softmax'))

    net.compile()
    net.fit(*data.get_train_val_data())
    net.save('./')

    net.load('./')
    score = net.evaluate(*data.get_test_data())
    print('Acc: ', score['acc'])

if __name__ == '__main__':
    main()
