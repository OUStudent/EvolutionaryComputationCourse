import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor

f = pd.read_csv("Sunspots.csv")
y = np.asarray(df['Monthly Mean Total Sunspot Number'])
size = len(y)
# 50% of data for training
train_ind = int(size * 0.50)
# 25% of data for validation and other 25% for testing
val_ind = int(size * 0.75)



# testing Genetic Programming algorithm using gplearn

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

    function_set = ['add', 'sub', 'mul', 'div', 'sin']
    temp_val = []
    temp_models = []
    for i in range(0, 4):
        gp = SymbolicRegressor(population_size=200, metric='mse',
                               generations=200, stopping_criteria=0.01,
                               init_depth=(4, 10), verbose=1, n_jobs=1, function_set=function_set,
                               const_range=(-100, 100))

        gp.fit(x_train, y_train)
        predictions = gp.predict(x_val)
        mse1 = mean_squared_error(y_val, predictions)
        print("  MSE Val: " + str(mse1))
        temp_val.append(mse1)
        temp_models.append(gp)
    best_index = np.argmin(temp_val)
    best_models.append(temp_models[best_index])
    best_fits.append(temp_val[best_index])

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
mse_test = mean_squared_error(y_test, best_model.predict(x_test))
#mse_test = fitness_function([best_model], [x_test, y_test])
print("\nBest Validation Fitness Values Per Window Size:")
index = 0
for fit in best_fits:
    print("Window Size: {} - Validation MSE: {}".format(index+min_window, best_fits[0]))
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
