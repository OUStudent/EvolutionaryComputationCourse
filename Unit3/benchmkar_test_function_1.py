import numpy as np
import matplotlib.pyplot as plt


def scale_fitness_1(x):
    return x + np.abs(np.min(x))


def scale_fitness_2(x):
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


def random_selection(n):
    return np.random.choice(range(0, n), n, replace=True)


def tournament_selection(fit, tourn_size, n):
    selection = []
    for i in range(0, n):
        competitors = np.random.choice(range(0, n), tourn_size, replace=False)
        best_index = np.argmax(fit[competitors])
        selection.append(competitors[best_index])
    return selection


# p1 and p2 are parents
# const_cross is gamma value
def crossover_method_1(p1, p2, const_cross):
    child = np.copy(p1)
    for j in range(0, np.shape(p1)[0]):
        child[j] = (1 - const_cross) * p1[j] + const_cross * p2[j]
    return child


def crossover_method_2(p1, p2):
    child = np.copy(p1)
    n = np.shape(p1)[0]
    random_nums = np.random.randint(low=0, high=1, size=n)
    if random_nums[0] == 0:
        child[0] = p2[0]

    elif random_nums[0] == 1:
        child[0] = p1[0]

    if n == 2:  # if dim==2, ensure even crossover
        if random_nums[0] == 0:
            child[1] = p1[1]
        elif random_nums[0] == 1:
            child[1] = p2[1]
    else:
        for j in range(1, n):
            if random_nums[j] == 0:
                child[j] = p1[j]
            else:
                child[j] = p2[j]
    return child


# const_mutate is the max value to mutate by
# Because the bounds could be different for each variable
# the const_mutate for each variable will be different so it
# is actually a list of values
def mutation_method_1(child, const_mutate):
    for j in range(0, np.shape(child)[0]):
        random_nums = np.random.uniform(-const_mutate[j], const_mutate[j], 1)
        child[j] = child[j] + random_nums[0]


# p1 and p2 are the parents
# f1 and f2 are the fitness values of parents
# p_cross and p_mutate are prob
# const_mutate is a list of max values to mutate by
# rep_method is int denoting reproduction type
def reproduction(p1, p2, f1, f2, p_cross, p_mutate, const_mutate, rep_method):
    p = np.random.uniform(0, 1, 2)
    if p[0] <= p_cross:
        if rep_method == 1:
            c_cross = np.random.normal(0.5, 0.15, 1)
            child = crossover_method_1(p1, p2, c_cross)
        elif rep_method == 2:
            child = crossover_method_2(p1, p2)
    else:
        if f1 < f2:
            child = np.copy(p2)
        else:
            child = np.copy(p1)

    if p[1] <= p_mutate:
        mutation_method_1(child, const_mutate)

    return child


# bounds is a list of two lists, first index is a list of the upper bounds for each variable,
# and second index is a list of the lower bounds for each variable
# In our example: x1 is [-2, 2] and x2 is [-1, 1], so bounds
# is equal to [ [2, 1], [-2, -1] ]
def evolve(init_gen, p_cross, p_mutate, bounds, fitness_function, mutate_bound=0.1, sel_method=1, rep_method=1,
           tourn_size=.10, elitism=0.0, max_iter=100, info=True, find_max=False):

    gen = np.copy(init_gen)
    # these next three lists will store the fitness values per generation so that we can plot them
    best_fit_values = []
    best_fitness = []
    mean_fitness = []

    upper_bound = bounds[0]
    lower_bound = bounds[1]
    bound_mut = mutate_bound * (np.abs(lower_bound) + np.abs(upper_bound))
    n, c = np.shape(gen)
    # tourn_size as an argument is a percentage, we convert it to an actual value
    # if tourn_size is too low, we make it equal to 1
    tourn_size = np.maximum(1, int(n * tourn_size))

    if elitism == 0.0:  # not using elitism
        elitism = -1
    else:  # set elitism best index
        elitism = int(n - np.ceil(n * elitism))

    for k in range(0, max_iter):
        fitness = fitness_function(gen)
        fit_mean = np.mean(fitness)
        if find_max:
            fit_best = np.max(fitness)
            best_index = np.argmax(fitness)
        else:
            fit_best = np.min(fitness)
            best_index = np.argmin(fitness)
        best_fit_values.append(gen[best_index,])
        best_fitness.append(fit_best)
        mean_fitness.append(fit_mean)
        if info:
            msg = "GENERATION {}:\n" \
                  "  Best Fit: {}, Mean Fit: {}".format(k, fit_best, fit_mean)
            print(msg)

        scaled_fit = scale_fitness_1(fitness)
        if not find_max:  # if find minimum scale using second method as well
            scaled_fit = scale_fitness_2(scaled_fit)

        ind = range(0, n)
        if sel_method == 1:
            fit_sum = np.sum(scaled_fit)
            fit = scaled_fit / fit_sum
        else:
            # if we aren't using proportional selection, no need to make fit equal to distribution
            fit = scaled_fit

        temp = np.column_stack((fit, ind))
        temp = temp[np.argsort(temp[:, 0])]

        if elitism != -1:  # if the elitism
            best_index = elitism
            best_ind = range(best_index, n)
            best_ind = np.array(temp[best_ind, 1], dtype=int).tolist()

        if sel_method == 1:
            cumulative_sum = np.cumsum(temp[:, 0])
            selection = roulette_wheel_selection(cumulative_sum, n)
        elif sel_method == 2:
            selection = tournament_selection(fit, tourn_size, n)
        elif sel_method == 3:
            selection = random_selection(n)

        mates = np.random.choice(selection, n, replace=False)

        children = []
        for i in range(0, n):
            children.append(reproduction(gen[selection[i], :], gen[mates[i], :], fit[selection[i]], fit[mates[i]],
                                         p_cross, p_mutate, bound_mut, rep_method))

        gen_next = np.asarray(children)
        # check bounds so that individuals stay in domain
        if bounds is not None:
            for i in range(0, n):
                for j in range(0, c):
                    if gen_next[i][j] > upper_bound[j]:
                        gen_next[i][j] = upper_bound[j]
                    elif gen_next[i][j] < lower_bound[j]:
                        gen_next[i][j] = lower_bound[j]

        if elitism != -1:
            fit_next = fitness_function(gen_next)
            ind = range(0, n)
            temp = np.column_stack((fit_next, ind))
            if find_max:
                temp = temp[np.argsort(-temp[:, 0])]
            else:
                temp = temp[np.argsort(temp[:, 0])]

            ind_replace = np.array(temp[best_index:n, 1], dtype=int).tolist()
            gen_next[ind_replace] = gen[best_ind]

        gen = gen_next

    x_range = range(0, max_iter)
    plt.plot(x_range, mean_fitness, label="Mean Fitness")
    plt.plot(x_range, best_fitness, label="Best Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Value")
    plt.suptitle("Mean and Best Fitness for Algorithm: [s,r,e,t]: [{},{},{},{}]".format(sel_method, rep_method,
                                                                                        elitism, tourn_size))
    plt.legend()
    plt.show()
    return best_fit_values


def six_hump_camel_back(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    return (4 - 2.1 * np.power(x1, 2) + np.power(x1, 4) / 3) * np.power(x1, 2) + x1 * x2 + (
                -4 + 4 * np.power(x2, 2)) * np.power(x2, 2)


size = 10
init_gen = np.empty(shape=(size, 2))
init_gen[:, 0] = np.random.uniform(-2, 2, size)
init_gen[:, 1] = np.random.uniform(-1, 1, size)

bounds = [[2, 1], [-2, -1]]
best_individuals = evolve(init_gen=init_gen, p_cross=0.25, p_mutate=0.25, bounds=bounds, mutate_bound=0.01, elitism=0.1,
                          fitness_function=six_hump_camel_back, rep_method=2, sel_method=3, tourn_size=.20, max_iter=30,
                          find_max=False)
print(best_individuals)
