import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, BatchNormalization, Flatten, Dense, \
    Activation
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import os
import psutil
from keras.utils import np_utils
import gc
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.backend import clear_session


def crossover_method_1(par):
    return np.mean(par, axis=0)


def crossover_method_2(par):
    child = []
    rand = np.random.randint(0, 2, len(par[0]))
    for i in range(0, len(rand)):
        if rand[i] == 0:
            child.append(par[0][i])
        else:
            child.append(par[1][i])
    return np.asarray(child)

class StateSave:

    def __init__(self, gen_cnn, gen_deep, best_fit, mean_fit, epoch, other=None):
        self.gen_cnn = gen_cnn
        self.gen_deep = gen_deep
        self.best_fit = best_fit
        self.mean_fit = mean_fit
        self.epoch = epoch
        self.other = other


def differential_evolution(init_gen_cnn, init_gen_deep, fitness_function, bounds_cnn, bounds_deep, n_diff=1,
                           beta_method='static', max_iter=100,
                           rep_method=1, cross_method=1, state_save=2, state_reload=None):
    best_fit = []
    mean_fit = []
    n, m, c, = init_gen_cnn.shape
    start = 0
    if state_reload:
        state = pickle.load(open(state_reload, "rb"))
        start = state.epoch
        gen_cnn = np.copy(state.gen_cnn)
        gen_deep = np.copy(state.gen_deep)
        mean_fit = state.mean_fit
        best_fit = state.best_fit
    else:
        gen_cnn = np.copy(init_gen_cnn)
        gen_deep = np.copy(init_gen_deep)
    fit = fitness_function(gen_cnn, gen_deep, 32, 32, 3)
    #fit = np.random.uniform(0, 1, 20)
    #fit = [ 1, 23, 13, 2, 13, ]
    for k in range(start, max_iter):
        fit_mean = np.mean(fit)
        fit_best = np.max(fit)
        fit_worst = np.min(fit)
        mean_fit.append(fit_mean)
        best_fit.append(fit_best)
        msg = "GENERATION {}:\n" \
              "  Best Fit: {}, Mean Fit: {}, Worst Fit: {}".format(k, fit_best, fit_mean, fit_worst)
        print(msg)
        print(gen_cnn[np.argmax(fit)])
        print(gen_deep[np.argmax(fit)])
        if (k + 1) % state_save == 0:
            print("State Save")
            obj = StateSave(gen_cnn=gen_cnn, gen_deep=gen_deep, best_fit=best_fit,
                            mean_fit=mean_fit, epoch=k)
            pickle.dump(obj, open("state_save{}".format(k), "wb"))
        for i in range(0, n):
            par_cnn = gen_cnn[i]
            par_deep = gen_deep[i]
            beta = 0.5

            ind = np.random.choice(range(0, n), n_diff * 2+1, replace=False)
            if rep_method == 1:  # /rand/n/z
                target_cnn = gen_cnn[ind[2]]  #np.argmin(fit)
                unit_cnn = mutation_1_n_z(target_cnn, gen_cnn[ind[0:2]], beta)
                target_deep = gen_deep[ind[2]]
                unit_deep = mutation_1_n_z(target_deep, gen_deep[ind[0:2]], beta)
  

            for l in range(0, len(unit_cnn)):
                for j in range(0, len(unit_cnn[l])):
                    if unit_cnn[l][j] > bounds_cnn[0][j]:
                        unit_cnn[l][j] = bounds_cnn[0][j]
                    elif unit_cnn[l][j] < bounds_cnn[1][j]:
                        unit_cnn[l][j] = bounds_cnn[1][j]
            for l in range(0, len(unit_deep)):
                for j in range(0, len(unit_deep[l])):
                    if unit_deep[l][j] > bounds_deep[0][j]:
                        unit_deep[l][j] = bounds_deep[0][j]
                    elif unit_deep[l][j] < bounds_deep[1][j]:
                        unit_deep[l][j] = bounds_deep[1][j]

            if cross_method == 1:
                child_cnn = crossover_method_1([par_cnn, unit_cnn])
                child_deep = crossover_method_1([par_deep, unit_deep])
            else:
                child_cnn = crossover_method_2([par_cnn, unit_cnn])
                child_deep = crossover_method_2([par_deep, unit_deep])
            f = fitness_function(child_cnn, child_deep, 32, 32, 3, one=True)
            if f > fit[i]:
                fit[i] = f
                gen_cnn[i] = child_cnn
                gen_deep[i] = child_deep


    return gen_cnn[np.argmax(fit)], gen_deep[np.argmax(fit)], gen_cnn, gen_deep, mean_fit, best_fit


(trainX, trainy), (testX, testy) = cifar10.load_data()


data_gen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True
            )
it = data_gen.flow(np.expand_dims(trainX[7], 0), batch_size=1)

for i in range(16):
    plt.subplot(4, 4, i+1)
    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
    plt.axis("Off")

plt.suptitle("Example of Augmentation", fontsize=18)
plt.show()
print(it)

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(trainX[i])
    plt.axis("off")

plt.suptitle("Sample of Original Images", fontsize=18)
plt.show()

trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX = trainX / 255
testX = testX / 255
trainy = np_utils.to_categorical(trainy, 10)
testy = np_utils.to_categorical(testy, 10)

print("Here")

def fitness_function(gen_cnn, gen_deep, img_width, img_height, img_channel, activation='relu',
                     num_output=10, max_epoch=100, one=False):
    if one:
        model = Sequential()
        print("Training Model: CPU: {}, RAM: {}, Memory: {}".format(psutil.cpu_percent(),
                                                                    psutil.virtual_memory().percent,
                                                                    psutil.Process(
                                                                        os.getpid()).memory_info().rss / 1024 ** 2))

        callback = [EarlyStopping(monitor='loss', patience=5), EarlyStopping(monitor='val_loss', patience=5),
                    EarlyStopping(monitor='val_accuracy', patience=5)]
        try:
            for module in gen_cnn:
                if module[0] <= 1:
                    output_channel = 16
                elif module[0] <= 2:
                    output_channel = 32
                elif module[0] <= 3:
                    output_channel = 64
                elif module[0] <= 4:
                    output_channel = 128
                elif module[0] <= 5:
                    output_channel = 256

                model.add(
                    Conv2D(output_channel, (3, 3),padding="same",
                           input_shape=(img_width, img_height, img_channel)))

                if module[1] >= 0:
                    if module[2] >= 0:
                        model.add(AveragePooling2D(pool_size=(2, 2)))
                    else:
                        model.add(MaxPooling2D(pool_size=(2, 2)))
                if module[3] >= 0:
                    model.add(BatchNormalization())
                if module[4] >= 0:
                    model.add(Activation(activation))
                if module[5] >= 0:
                    dropout = np.round(module[6], 2)
                    model.add(Dropout(dropout))

            model.add(Flatten())

            for module in gen_deep:
                node = np.round(module[0], 0)
                model.add(Dense(node, activation=activation))
                if module[1] >= 0:
                    model.add(BatchNormalization())
                if module[2] >= 0:
                    dropout = np.round(module[3], 2)
                    model.add(Dropout(dropout))

            model.add(Dense(num_output))
            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            history = model.fit(trainX[0:20000], trainy[0:20000], epochs=max_epoch, verbose=0, callbacks=callback,
                                batch_size=128, validation_data=(trainX[20000:25000], trainy[20000:25000]))
            fit = np.nanmax(history.history['val_accuracy'])
            #print("Val Fit: {}".format(fit))
        except Exception as e:
            print("Model Architecture Failure")
            print(e)
            fit = -1
        del model
        gc.collect()
        clear_session()
        tf.compat.v1.reset_default_graph()
        return fit

    fits = []
    histories = []
    for chromosome_cnn, chromosome_deep in zip(gen_cnn, gen_deep):
        print("Training Model: CPU: {}, RAM: {}, Memory: {}".format(psutil.cpu_percent(),
                                                                    psutil.virtual_memory().percent,
                                                                    psutil.Process(
                                                                        os.getpid()).memory_info().rss / 1024 ** 2))
        model = Sequential()
        callback = [EarlyStopping(monitor='loss', patience=10), EarlyStopping(monitor='val_loss', patience=10),
                    EarlyStopping(monitor='val_accuracy', patience=10)]
        try:
            
            for module in chromosome_cnn:

                if module[0] <= 1:
                    output_channel = 16
                elif module[0] <= 2:
                    output_channel = 32
                elif module[0] <= 3:
                    output_channel = 64
                elif module[0] <= 4:
                    output_channel = 128
                elif module[0] <= 5:
                    output_channel = 256

                model.add(
                    Conv2D(output_channel, (3, 3), padding="same",
                           input_shape=(img_width, img_height, img_channel)))

                if module[1] >= 0:
                    if module[2] >= 0:
                        model.add(AveragePooling2D(pool_size=(2, 2)))
                    else:
                        model.add(MaxPooling2D(pool_size=(2, 2)))
                if module[3] >= 0:
                    model.add(BatchNormalization())
                if module[4] >= 0:
                    model.add(Activation(activation))
                if module[5] >= 0:
                    dropout = np.round(module[6], 2)
                    model.add(Dropout(dropout))

            model.add(Flatten())

            for module in chromosome_deep:
                node = np.round(module[0], 0)
                model.add(Dense(node, activation=activation))
                if module[1] >= 0:
                    model.add(BatchNormalization())
                if module[2] >= 0:
                    dropout = np.round(module[3], 2)
                    model.add(Dropout(dropout))
            
         
            print(model.summary())
            history = model.fit(trainX[0:20000], trainy[0:20000], batch_size=128, epochs=max_epoch,
                                verbose=1, callbacks=callback, validation_data=(trainX[20000:25000], trainy[20000:25000]))
            fit = np.nanmax(history.history['val_accuracy'])
            scores = model.evaluate(testX, testy, verbose=1)
            histories.append(history.history)
            print(scores)
            print("Val Fit: {}".format(fit))
            fits.append(fit)
        except Exception as e:
            print("Model Architecture Failure")
            print(e)
            fits.append(-1)
        del model
        gc.collect()
        clear_session()
        tf.compat.v1.reset_default_graph()
    return np.asarray(fits)


upper_bound_cnn = [5, 1, 1, 1, 1, 1, 0.5]
lower_bound_cnn = [0, -1, -1, -1, -1, -1, 0.2]
bounds_cnn = [upper_bound_cnn, lower_bound_cnn]
size = 20
cnn_module = 4
init_gen_cnn = np.empty(shape=(size, cnn_module, len(upper_bound_cnn)))
for i in range(0, size):
    for j in range(0, cnn_module):
        for k in range(0, len(upper_bound_cnn)):
            init_gen_cnn[i, j, k] = np.random.uniform(lower_bound_cnn[k], upper_bound_cnn[k], 1)[0]

upper_bound_deep = [250, 0.5, 0.5, 0.5]
lower_bound_deep = [50, -1, -1, 0.2]
bounds_deep = [upper_bound_deep, lower_bound_deep]
deep_module = 2
init_gen_deep = np.empty(shape=(size, deep_module, len(upper_bound_deep)))
for i in range(0, size):
    for j in range(0, deep_module):
        for k in range(0, len(upper_bound_deep)):
            init_gen_deep[i, j, k] = np.random.uniform(lower_bound_deep[k], upper_bound_deep[k], 1)[0]


best_cnn, best_deep, gen_cnn, gen_deep, mean_fit, best_fit = differential_evolution(init_gen_cnn=init_gen_cnn,
                                                                                    init_gen_deep=init_gen_deep,
                                                                                    fitness_function=fitness_function,
                                                                                    bounds_cnn=bounds_cnn,
                                                                                    bounds_deep=bounds_deep,
                                                                                    max_iter=50, cross_method=2)

obj = StateSave(gen_cnn=gen_cnn, gen_deep=gen_deep, best_fit=best_fit,
                mean_fit=mean_fit, epoch=50)
pickle.dump(obj, open("final", "wb"))
