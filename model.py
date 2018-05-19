from keras import optimizers, losses, activations, models
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, Flatten, BatchNormalization,\
    GRU, Activation


def get_model(input_shape, classes):
    inp = Input(shape=input_shape)

    img_1 = Convolution1D(16, kernel_size=9, activation=activations.relu, padding="valid")(inp)
    img_1 = Convolution1D(16, kernel_size=9, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=16)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=4)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=4)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(64, activation=activations.relu)(img_1)
    dense_1 = Dense(1028, activation=activations.relu)(dense_1)
    dense_1 = Dense(classes, activation=activations.softmax)(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.0001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()

    return model


def get_model2(input_shape, classes):
    inp = Input(shape=input_shape)
    img_1 = Convolution1D(16, kernel_size=2, activation=activations.relu, padding="valid")(inp)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Convolution1D(32, kernel_size=9, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=4)(img_1)
    img_1 = Convolution1D(32, kernel_size=9, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=4)(img_1)
    img_1 = Flatten()(img_1)
    dense_1 = Dense(500, activation=activations.relu)(img_1)
    dense_1 = Dropout(rate=0.5)(dense_1)
    dense_1 = Dense(classes, activation=activations.softmax)(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.0001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()

    return model


def get_model3(input_shape, classes):
    inp = Input(shape=input_shape)
    # X = Convolution1D(196, kernel_size=15, strides=4)(inp)
    x = Convolution1D(64, kernel_size=15, strides=4)(inp)
    x = BatchNormalization()(x)  # Batch normalization
    x = Activation('relu')(x)  # ReLu activation
    x = Dropout(0.5)(x)  # dropout (use 0.8)
    # X = GRU(units=128, return_sequences=True)(X)
    x = GRU(units=16, return_sequences=True)(x)
    x = Dropout(0.8)(x)
    x = BatchNormalization()(x)
    # X = GRU(units=128, return_sequences=True)(X)
    x = GRU(units=16, return_sequences=True)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(classes, activation=activations.softmax)(x)
    # X = TimeDistributed(Dense(nclass, activation="sigmoid"))(X)

    model = models.Model(inputs=inp, outputs=x)
    opt = optimizers.Adam(0.0001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    model.summary()

    return model
