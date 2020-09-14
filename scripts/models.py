import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

keras = tf.keras
RANDOM_SEED = 42
keras.backend.clear_session()
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

#Single Layer LSTM model with Validation Callback
def gen_LSTM1_model(train,val,ext,epochs=50,batch_size=16,patience=10,units=120,num_outputs=1):
    #Defining the Network
    model = keras.Sequential([
        keras.layers.LSTM(
            units=units,
            input_shape=(train[0].shape[1], train[0].shape[2])
        ),
        keras.layers.Dense(num_outputs)
    ])
    # Compiling the Network
    optimizer = keras.optimizers.Adam(0.001)
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=optimizer
    )
    early_stopping = keras.callbacks.EarlyStopping(patience=patience)
    #Training
    history = model.fit(
        train[0],
        train[1],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=val,
        callbacks=[early_stopping]
    )
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    model.save("saved_models/" + "LSTM1_E" + str(epochs)+"-"+ext+".h5")

def gen_CNN1_LSTM1_model(train,val,ext,epochs=50,batch_size=16,patience=10,units=120,filters=32, kernel_size=2,num_outputs=1):
    #Defining the Network
    model = keras.Sequential([
        keras.layers.Conv1D(
            filters=32,
            kernel_size = kernel_size,
            strides=1,
            padding="causal",
            activation='relu',
            input_shape=(train[0].shape[1], train[0].shape[2])
        ),
        keras.layers.LSTM(
            units=units
        ),
        keras.layers.Dense(num_outputs)
    ])
    # Compiling the Network
    optimizer = keras.optimizers.Adam(0.001)
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=optimizer
    )
    early_stopping = keras.callbacks.EarlyStopping(patience=patience)
    #Training
    history = model.fit(
        train[0],
        train[1],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=val,
        callbacks=[early_stopping]
    )
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    model.save("saved_models/" + "CNN1LSTM1_E" + str(epochs)+"-"+ext+".h5")

def gen_CNN1_Dense1_LSTM1_model(train,val,ext,epochs=50,batch_size=16,patience=10,units=120,filters=32, kernel_size=2,dense_units=512,num_outputs=1):
    #Defining the Network
    model = keras.Sequential([
        keras.layers.Conv1D(
            filters=32,
            kernel_size = kernel_size,
            strides=1,
            padding="causal",
            activation='relu',
            input_shape=(train[0].shape[1], train[0].shape[2])
        ),
        keras.layers.Dense(
            units = dense_units
        ),
        keras.layers.Dropout(
            rate=.4
        ),
        keras.layers.LSTM(
            units=units
        ),
        keras.layers.Dense(num_outputs)
    ])
    # Compiling the Network
    optimizer = keras.optimizers.Adam(0.001)
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=optimizer
    )
    early_stopping = keras.callbacks.EarlyStopping(patience=patience)
    #Training
    history = model.fit(
        train[0],
        train[1],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=val,
        callbacks=[early_stopping]
    )
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    model.save("saved_models/" + "CNN1_Dense1_LSTM1_E" + str(epochs)+"-"+ext+".h5")

def gen_CNN1_RNN1_model(train,val,ext,epochs=50,batch_size=16,patience=10,units=120,filters=32, kernel_size=2,dense_units=512,num_outputs=1):
    #Defining the Network
    model = keras.Sequential([
        keras.layers.Conv1D(
            filters=32,
            kernel_size = kernel_size,
            strides=1,
            padding="causal",
            activation='relu',
            input_shape=(train[0].shape[1], train[0].shape[2])
        ),
        keras.layers.SimpleRNN(
            units=units
        ),
        keras.layers.Dense(num_outputs)
    ])
    # Compiling the Network
    optimizer = keras.optimizers.Adam(0.001)
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=optimizer
    )
    early_stopping = keras.callbacks.EarlyStopping(patience=patience)
    #Training
    history = model.fit(
        train[0],
        train[1],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=val,
        callbacks=[early_stopping]
    )
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    model.save("saved_models/" + "CNN1_RNN1_E" + str(epochs)+"-"+ext+".h5")

def gen_CNN5_model(train,val,ext,epochs=50,batch_size=16,patience=10,units=120,filters=32, kernel_size=2,dense_units=512,num_outputs=1):
    #Defining the Network
    model = keras.Sequential([
        keras.layers.Conv1D(
            filters=32,
            kernel_size = kernel_size,
            strides=1,
            padding="causal",
            activation='relu',
            input_shape=(train[0].shape[1], train[0].shape[2])
        ),
        keras.layers.Conv1D(
            filters=64,
            kernel_size = kernel_size*2,
            strides=1,
            padding="causal",
            activation='relu',
        ),
        keras.layers.Conv1D(
            filters=128,
            kernel_size = kernel_size*3,
            strides=1,
            padding="causal",
            activation='relu',
        ),
        keras.layers.Dense(num_outputs)
    ])
    # Compiling the Network
    optimizer = keras.optimizers.Adam(0.001)
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=optimizer,
        metrics=['mean_squared_error']
    )
    early_stopping = keras.callbacks.EarlyStopping(patience=patience)
    #Training
    history = model.fit(
        train[0],
        train[1],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=val,
        callbacks=[early_stopping]
    )
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('model accuracy')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    model.save("saved_models/" + "CNN5_E" + str(epochs)+"-"+ext+".h5")