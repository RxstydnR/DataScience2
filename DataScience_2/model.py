from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten,Activation
from keras.layers.convolutional import Conv1D, UpSampling1D
from keras.layers.pooling import MaxPooling1D

from keras.layers import SimpleRNN,LSTM,GRU,RNN


def get_1DCNN(sequence_len,n_feature):
    
    model = Sequential()
    
    model.add(Conv1D(64, 8, padding='same', input_shape=(sequence_len,n_feature), activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Conv1D(64, 8, padding='same', activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Conv1D(32, 8, padding='same', activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    return model

def get_LSTM(sequence_len,n_feature):
    
    model = Sequential()
    model.add(LSTM(64,
                input_shape=(sequence_len,n_feature),
                return_sequences=True, recurrent_dropout = 0.2))
    model.add(LSTM(32,recurrent_dropout=0.2))
    model.add(Dense(1))
    

    return model

def get_LSTM_2(sequence_len,n_feature):
    
    model = Sequential()
    model.add(LSTM(64,input_shape=(sequence_len,n_feature)))
    model.add(Dense(1))
    

    return model

def get_RNN(sequence_len,n_feature):
    model = Sequential()
    model.add(SimpleRNN(64, input_shape=(sequence_len,n_feature),return_sequences=True))
    model.add(SimpleRNN(32))#,return_sequences=True))
    #model.add(SimpleRNN(16))
    #model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    

    return model

def get_RNN_2(sequence_len,n_feature):
    model = Sequential()
    model.add(SimpleRNN(64, input_shape=(sequence_len,n_feature)))
    model.add(Dense(1))
    
    return model


def get_CNN_LSTM(sequence_len,n_feature):
    model = Sequential()
    
    # CNN
    model.add(Conv1D(128, 3, padding='same', input_shape=(sequence_len,n_feature), activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Conv1D(64, 3, padding='same', activation='relu'))
    model.add(Conv1D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))

    model.add(LSTM(64,
                return_sequences=True, recurrent_dropout = 0.2))
    model.add(LSTM(32))
    model.add(Dense(1))
    

    return model


def get_CNN_LSTM_2(sequence_len,n_feature):
    model = Sequential()
    
    # CNN
    model.add(Conv1D(64, 8, padding='same', input_shape=(sequence_len,n_feature), activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Conv1D(32, 8, padding='same', activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(LSTM(64))
    model.add(Dense(1))
    

    return model

def get_CNN_LSTM_3(sequence_len,n_feature):
    model = Sequential()
    # CNN
    model.add(Conv1D(64, 8, padding='same', input_shape=(sequence_len,n_feature), activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Conv1D(64, 8, padding='same', activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Conv1D(32, 8, padding='same', activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))

    model.add(LSTM(64,
                return_sequences=True, recurrent_dropout = 0.2))
    model.add(LSTM(32))
    model.add(Dense(1))
    

    return model

def get_CNN_LSTM_6(sequence_len,n_feature):
    model = Sequential()
    # CNN
    model.add(Conv1D(64, 8, padding='same', input_shape=(sequence_len,n_feature), activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Conv1D(64, 8, padding='same', activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Conv1D(32, 8, padding='same', activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))

    model.add(LSTM(64,
                return_sequences=True, recurrent_dropout = 0.2))
    model.add(LSTM(32))
    model.add(Dense(32))
    model.add(Dense(1))
    

    return model


def get_CNN_LSTM_5(sequence_len,n_feature):
    model = Sequential()
    # CNN
    model.add(Conv1D(64, 8, padding='same', input_shape=(sequence_len,n_feature), activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Conv1D(64, 8, padding='same', activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Conv1D(32, 8, padding='same', activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))

    model.add(LSTM(64))
    model.add(Dense(1))
    

    return model

def get_CNN_LSTM_4(sequence_len,n_feature):
    model = Sequential()
    # CNN
    model.add(Conv1D(64, 8, padding='same', input_shape=(sequence_len,n_feature), activation='relu'))
    model.add(Conv1D(64, 8, padding='same', activation='relu'))
    model.add(Conv1D(32, 8, padding='same', activation='relu'))

    model.add(LSTM(64,
                return_sequences=True, recurrent_dropout = 0.2))
    model.add(LSTM(32))
    model.add(Dense(1))
    

    return model


def get_CNN_RNN(sequence_len,n_feature):
    model = Sequential()
    
    # CNN
    model.add(Conv1D(128, 3, padding='same', input_shape=(sequence_len,n_feature), activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Conv1D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Conv1D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))

    model.add(SimpleRNN(64,
                return_sequences=True, recurrent_dropout = 0.2))
    model.add(SimpleRNN(32))
    model.add(Dense(1))
    

    return model

def get_CNN_RNN_2(sequence_len,n_feature):
    model = Sequential()
    
    # CNN
    model.add(Conv1D(64, 3, padding='same', input_shape=(sequence_len,n_feature), activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Conv1D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Conv1D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))

    model.add(SimpleRNN(64,
                return_sequences=True, recurrent_dropout = 0.2))
    model.add(SimpleRNN(32))
    model.add(Dense(1))
    

    return model

def get_CNN_RNN_3(sequence_len,n_feature):
    model = Sequential()
    
    # CNN
    model.add(Conv1D(64, 3, padding='same', input_shape=(sequence_len,n_feature), activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Conv1D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Conv1D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))

    model.add(SimpleRNN(64))
    model.add(Dense(1))
    

    return model