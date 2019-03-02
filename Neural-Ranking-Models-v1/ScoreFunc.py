import numpy as np
from keras.models import Sequential 
from keras.layers import Dense, Input

np.random.seed(1337)

def creatModel(input_dim):
    input_layer = Input(shape = (input_dim,), name="input_layer")
    h1 = Dense(units=128, activation="relu")(input_layer)
    h2 = Dense(units=128, activation="relu")(h1)
    h3 = Dense(units=128, activation="relu")(h2)
    predicts = Dense(1, activation='linear')(h3)
    model = Model(input=[input_layer], output=predicts)
    return model

if __name__ == "__main__":    
    input_dim=50
    batch_size=100
    epochs=20
    
    pred_train = np.random.rand(10000, input_dim)
    label_train = np.sum(pred_train, axis=1)
    pred_test = np.random.rand(10000, input_dim)
    label_test = np.sum(pred_test, axis=1)
    
    model = creatModel(input_dim)
    model.compile(loss='mse', optimizer='adam', metrics=['acc'])
    model.summary()
    model.fit(pred_train, label_train, batch_size=batch_size, epochs=epochs)
    
    result = model.evaluate(pred_test, label_test)
    print (result)
