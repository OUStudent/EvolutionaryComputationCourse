import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(x, 0)

# models represents the list of networks
# data is composed of the time series input and output
def fitness_function(models, data):
    mse_values = []
    x = data[0]
    y = data[1]
    for network in models:
        predictions = network.predict(x)
        mse = mean_squared_error(y, predictions)
        mse_values.append(mse)
    return np.asarray(mse_values)


def scale_fitness(x):
    return 1 / (1 + x)


def roulette_wheel_selection(cumulative_sum, n):
    ind = []
    r = np.random.uniform(0, 1, n)
    for i in range(0, n):
        index = 0
        while cumulative_sum[index] < r[i]:
            index += 1
        ind.append(index)
    return ind

# p1 and p2 are parents
# const_cross is the coefficient for the linear combination, ranges between [0, 1]
# if near 0, it will favor p1; if near 1, it will favor p2; if equal to 0.5, it is the mean between the parents
def crossover(p1, p2, const_cross):
    # initialize new network with empty layer weights and biases
    child = EvolvableNetwork(layer_nodes=p1.layer_nodes, num_input=p1.num_input, num_output=p1.num_output,
                             initialize=False)
    # fill child weight and bias matrices from the parents
    for i in range(0, p1.layer_count+1):
        child.layer_weights.append( (1-const_cross)*p1.layer_weights[i]+const_cross*p2.layer_weights[i])
        child.layer_biases.append( (1-const_cross)*p1.layer_biases[i]+const_cross*p2.layer_biases[i])

    return child

# const_mutate is the max value to mutate by
def mutation(child, const_mutate):
    # loop over all layers
    for i in range(0, child.layer_count+1):
        n, c = child.layer_weights[i].shape
        # these are the random weights to add the current child
        r_w = np.random.uniform(-const_mutate, const_mutate, n*c)
        # loop over all rows and columns for the current layer
        for nr in range(0, n):
            for nc in range(0, c):
                child.layer_weights[i][nr, nc] += r_w[nr*c+nc]

    # loop over all layers
    for i in range(0, child.layer_count+1):
        c = child.layer_biases[i].shape[0]
        # these are the random weights to add the current child
        r_w = np.random.uniform(-const_mutate, const_mutate, c)
        # loop over all columns of the vector
        for nc in range(0, c):
            child.layer_biases[i][nc] += r_w[nc]

# p1 and p2 are parents
# const_mutate is the max value to mutate by
def reproduce(p1, p2, const_mutate, fitness_function, train_data):
    # create a different gamma coefficient for averaging
    # crossover for each offspring
    c_cross = np.random.normal(0.5, 0.15, 4)
    ch1 = crossover(p1, p2, c_cross[0])
    ch2 = crossover(p1, p2, c_cross[1])
    ch3 = crossover(p1, p2, c_cross[2])
    ch4 = crossover(p1, p2, c_cross[3])
    # mutate only two of the individuals
    mutation(ch3, const_mutate)
    mutation(ch4, const_mutate)
    # pool offspring with parents
    all = [p1, p2, ch1, ch2, ch3, ch4]
    fit = fitness_function(all, train_data)
    # return the individual with the min fitness value
    return all[np.argmin(fit)]

class EvolvableNetwork:

    # Layer Nodes is a list of int values denoting the number of nodes per layer
    # For example, if layer_nodes = [3, 5, 3], we have three hidden layers with a 3-5-3 node architecture
    # num_input and num_output refer to the number of input and output variables
    # I will explain the purpose of the Initialize boolean later, but simply if False, do not create the weight
    # and bias matrices
    def __init__(self, layer_nodes, num_input, num_output, initialize=True):
        self.layer_count = len(layer_nodes)
        self.layer_nodes = layer_nodes
        self.num_input = num_input
        self.num_output = num_output
        self.activation_function = relu

        self.layer_weights = []
        self.layer_biases = []

        if not initialize:  # I will discuss the purpose of this later
            return

        # create the NxM weight and bias matrices for input Layer
        self.layer_weights.append(
            np.random.uniform(-1, 1, num_input * layer_nodes[0]).reshape(num_input, layer_nodes[0]))
        self.layer_biases.append(np.random.uniform(-1, 1, layer_nodes[0]))

        # create the weight matrices for Hidden Layers
        for i in range(1, self.layer_count):
            self.layer_weights.append(
                np.random.uniform(-1, 1, layer_nodes[i-1]*layer_nodes[i]).reshape(layer_nodes[i-1], layer_nodes[i]))
            self.layer_biases.append(np.random.uniform(-1, 1, layer_nodes[i]).reshape(1, layer_nodes[i]))

        # Create the weight and bias matrices for output Layer
        self.layer_weights.append(
            np.random.uniform(-1, 1, layer_nodes[self.layer_count-1]*num_output).reshape(layer_nodes[self.layer_count-1],
                                                                                         num_output))
        self.layer_biases.append(np.random.uniform(-1, 1, num_output).reshape(1, num_output))

    def predict(self, x):  # same as forward pass, performs matrix multiplication of the weights
        output = self.activation_function(np.dot(x, self.layer_weights[0]) + self.layer_biases[0])
        for i in range(1, self.layer_count + 1):
            if i == self.layer_count:  # last layer so don't use activation function
                output = (np.dot(output, self.layer_weights[i]) + self.layer_biases[i])
            else:
                output = self.activation_function(
                    np.dot(output, self.layer_weights[i]) + self.layer_biases[i])
        if self.num_output == 1:  # if there is only one output variable then reshape
            return output.reshape(len(x), )
        return output

# const_mutate in our example is the actual max value to mutate by, not percentage
# use train and val data to prevent overfitting - we early stop if the mean of
# validation data increases for three straight iterations
def evolve(init_gen, const_mutate, max_iter, train_data, val_data):
    gen = init_gen
    mean_fitness = []
    val_mean = []  # validation mean value
    best_fitness = []
    prev_val = 1000
    n = len(gen)
    val_index = 0
    for k in range(0, max_iter):
        fitness = fitness_function(gen, train_data)
        # scale so that large values -> small
        # and small values -> large
        scaled_fit = scale_fitness(fitness)

        # create distribution for proportional selection
        fit_sum = np.sum(scaled_fit)
        fit = scaled_fit / fit_sum
        cumulative_sum = np.cumsum(fit)

        selected = roulette_wheel_selection(cumulative_sum, n)
        mates = roulette_wheel_selection(cumulative_sum, n)

        children = []
        for i in range(0, n):
            children.append(reproduce(gen[selected[i]], gen[mates[i]], const_mutate, fitness_function, train_data))

        gen_next = children

        # evaluate training data
        fit = fitness_function(gen_next, train_data)
        fit_mean = np.mean(fit)
        fit_best = np.min(fit)
        mean_fitness.append(fit_mean)
        best_fitness.append(fit_best)

        # evaluate validation data
        val_fit = fitness_function(gen_next, val_data)
        val_fit_mean = np.mean(val_fit)
        val_mean.append(val_fit_mean)

        print("Generation: " + str(k))
        print(" Best: {}, Avg: {}".format(fit_best, fit_mean))
        print(" Validation: {}".format(val_fit_mean))

        gen = gen_next

        # check if previous iteration validation increased or decreased
        if val_fit_mean > prev_val:
            val_index += 1
        else:
            val_index = 0
        if val_index == 3:  # val has increased for three straight iterations
            print("Over Fitting, Stopping...")
            break
        prev_val = val_fit_mean

    # use validation data to choose best model from current generation
    val_fit = fitness_function(gen_next, val_data)
    best_val = np.min(val_fit)
    best_ind = np.argmin(val_fit)
    print("Best Model: ")
    print(" Validation: {}".format(best_val))
    return gen_next[best_ind]




# compile the data 

df = pd.read_csv("Sunspots.csv")
y = np.asarray(df['Monthly Mean Total Sunspot Number'])
size = len(y)
# 50% of data for training
train_ind = int(size * 0.50)
# 25% of data for validation and other 25% for testing
val_ind = int(size * 0.75)



# testing Genetic Algorithm

max_window = 10
min_window = 3
initial_population_size = 100  # 10 neural networks
best_models = []  # best model from each run of the algorithm per window size
best_fits = []
# randomly shuffle data through indices:
shuffled_indices = np.asarray(range(0, size-max_window))
np.random.shuffle(shuffled_indices)
# loop over each window size
for vision in range(min_window, max_window + 1):
    input = []
    output = []
    # creates the window length size for each value
    # because the first couple values will not have
    # a full window we skip them, that's why start
    # at i and not 0
    for j in range(vision, size):
        input.append(y[(j - vision):j].tolist())
        output.append(y[j])

    input = np.asarray(input)
    output = np.asarray(output)

    temp = np.column_stack((output, input))

    # instead of shuffle each time here, we shuffle once outside loop
    # so that all window sizes have the same final array
    temp = temp[shuffled_indices]

    output = temp[:, 0]
    input = temp[:, 1:]

    y_train = output[0:train_ind]
    y_val = output[train_ind:val_ind]
    y_test = output[val_ind:size]
    x_train = input[0:train_ind]
    x_val = input[train_ind:val_ind]
    x_test = input[val_ind:size]

    init_gen = []
    for i in range(0, initial_population_size):
        init_gen.append(EvolvableNetwork(layer_nodes=[5, 5, 5], num_input=vision, num_output=1, initialize=True))

    best_model = evolve(init_gen, const_mutate=0.1, max_iter=200, train_data=[x_train, y_train], val_data=[x_val, y_val])
    best_models.append(best_model)
    best_fits.append(fitness_function([best_model], [x_val, y_val]))

# get best model
best_index = np.argmin(best_fits)
best_model = best_models[best_index]

# recreate data with that window size
vision = best_index + min_window
input = []
output = []
for j in range(vision, size):
    input.append(y[(j - vision):j].tolist())
    output.append(y[j])
input = np.asarray(input)
output = np.asarray(output)
temp = np.column_stack((output, input))
temp = temp[shuffled_indices]
output_2 = temp[:, 0]
input_2 = temp[:, 1:]
y_test = output_2[val_ind:size]
x_test = input_2[val_ind:size]

# evaluate test data
mse_test = fitness_function([best_model], [x_test, y_test])
print("\nBest Validation Fitness Values Per Window Size:")
index = 0
for fit in best_fits:
    print("Window Size: {} - Validation MSE: {}".format(index+min_window, best_fits[index][0]))
    index += 1
print("Validation Error: Mean w/ std: {}+-{}".format(np.mean(best_fits), np.std(best_fits)))
print("Best Model: \n"
      " Window Size : {}\n"
      " MSE for Test Data Set : {}".format(best_index+3, mse_test[0]))

# printing best model predictions
xaxis = range(vision, len(y))
plt.plot(xaxis, y[vision:], label="Actual")
plt.plot(xaxis, best_model.predict(input), label="Prediction")
plt.xlabel("Months")
plt.ylabel("Mean Total Sunspot Number")
plt.title("Sunspot Cycle From 1749-2021")
plt.legend()
plt.show()







# testing MLP REGRESSOR

best_fits = []
best_models = []
for vision in range(min_window, max_window + 1):
    input = []
    output = []
    for j in range(vision, size):
        input.append(y[(j - vision):j].tolist())
        output.append(y[j])

    input = np.asarray(input)
    output = np.asarray(output)

    temp = np.column_stack((output, input))
    temp = temp[shuffled_indices]

    output = temp[:, 0]
    input = temp[:, 1:]

    y_train = output[0:train_ind]
    y_val = output[train_ind:val_ind]
    y_test = output[val_ind:size]
    x_train = input[0:train_ind]
    x_val = input[train_ind:val_ind]
    x_test = input[val_ind:size]
    mlp = MLPRegressor(hidden_layer_sizes=[5, 5, 5], max_iter=10000, verbose=True,
                       learning_rate='adaptive', early_stopping=True)
    mlp.fit(x_train, y_train)
    predictions = mlp.predict(x_val)
    mse = mean_squared_error(y_val, predictions)
    best_models.append(mlp)
    best_fits.append(mse)

# get best model
best_index = np.argmin(best_fits)
best_model = best_models[best_index]

# recreate data with that window size
vision = best_index + min_window
input = []
output = []
for j in range(vision, size):
    input.append(y[(j - vision):j].tolist())
    output.append(y[j])
input = np.asarray(input)
output = np.asarray(output)
temp = np.column_stack((output, input))
temp = temp[shuffled_indices]
output_2 = temp[:, 0]
input_2 = temp[:, 1:]
y_test = output_2[val_ind:size]
x_test = input_2[val_ind:size]

# evaluate test data
predictions = best_model.predict(x_test)
mse_test = mean_squared_error(y_test, predictions)
print("\nBest Validation Fitness Values Per Window Size:")
index = 0
for fit in best_fits:
    print("Window Size: {} - Validation MSE: {}".format(index+min_window, best_fits[index]))
    index += 1

print("Validation Error: Mean w/ std: {}+-{}".format(np.mean(best_fits), np.std(best_fits)))
print("Best Model: \n"
      " Window Size : {}\n"
      " MSE for Test Data Set : {}".format(best_index+3, mse_test))

# printing best model predictions
xaxis = range(vision, len(y))
plt.plot(xaxis, y[vision:], label="Actual")
plt.plot(xaxis, best_model.predict(input), label="Prediction")
plt.xlabel("Months")
plt.ylabel("Mean Total Sunspot Number")
plt.title("Sunspot Cycle From 1749-2021")
plt.legend()
plt.show()
