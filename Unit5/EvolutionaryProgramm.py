import numpy as np
import matplotlib.pyplot as plt


# fitness function
def pressure_vessel(x):
    x1 = np.round(x[:, 0]/0.0625, 0)*0.0625
    x2 = np.round(x[:, 1]/0.0625, 0)*0.0625
    x3 = x[:, 2]
    x4 = x[:, 3]
    return 0.6224*x1*x3*x4+1.7781*x2*x3*x3+3.1661*x1*x1*x4+19.84*x1*x1*x3


# constraints
def g(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = x[:, 3]

    g1 = -x1+0.0193*x3

    g2 = -x2 + 0.00954*x3

    g3 = -np.pi*x3*x3*x4-(4*np.pi/3)*x3*x3*x3+1296000

    g4 = x4 - 240

    return np.asarray([g1, g2, g3, g4]).T


# calculate the number of times each individual broke a constraint
# and by how much
def constraints(x):
    gs = g(x)
    res = []
    # loop over each individual
    for ind in gs:

        count = 0  # number of times ind broke a constraint
        if ind[0] > 0:
            g1 = ind[0]  # broke constraint difference
            count += 1
        else:
            g1 = 0

        if ind[1] > 0:
            g2 = ind[1]  # broke constraint difference
            count += 1
        else:
            g2 = 0

        if ind[2] > 0:
            g3 = ind[2]  # broke constraint difference
            count += 1
        else:
            g3 = 0

        if ind[3] > 0:
            g4 = ind[3]  # broke constraint difference
            count += 1
        else:
            g4 = 0

        res.append([count, g1, g2, g3, g4])

    return np.asarray(res)


# mutates the current generation to create offspring using
# the strategy parameters sigma
def mutation(gen, sigma):
    n_x = len(gen[0])  # number of variables
    tau = 1/(np.sqrt(2*np.sqrt(n_x)))
    tau_prime = 1/(np.sqrt(2*n_x))
    offspring_values = []
    offspring_sigma = []
    # loop over each individual and associated strategy parameters
    for (parent, strategy) in zip(gen, sigma):
        r = np.random.normal(0, 1, len(parent))
        child_sigma = strategy * np.exp(tau * r + tau_prime * r)

        #r = np.random.laplace(0, 1, len(parent))
        r = np.random.normal(0, 1, len(parent))
        #r = np.random.standard_cauchy(len(parent))
        child_value = np.copy(parent)+child_sigma*r

        offspring_values.append(child_value)
        offspring_sigma.append(child_sigma)
    return np.asarray(offspring_values), np.asarray(offspring_sigma)


def crossover(p1, p2, s1, s2):
    child = np.copy(p1)
    sig = np.copy(s1)
    bits = np.random.randint(0, 2, len(child))
    for i in range(0, len(child)):
        if bits[i] == 0:
            child[i] = p2[i]
            sig[i] = s2[i]
    return child, sig


def reproduction(p1, p2, s1, s2):
    ch1_v, ch1_s = crossover(p1, p2, s1, s2)
    ch2_v, ch2_s = crossover(p1, p2, s1, s2)
    ch3_v, ch3_s = mutation([ch1_v], [ch1_s])
    ch4_v, ch4_s = mutation([ch2_v], [ch2_s])
    return [ch1_v, ch2_v, ch3_v[0], ch4_v[0]], [ch1_s, ch2_s, ch3_s[0], ch4_s[0]]


# fit is current individual's fitness
# tourn_fit is the fitness from the tournament
def calculate_relative_fit(fit, tourn_fit):
    s = 0
    r = np.random.uniform(0, 1, len(tourn_fit))
    for i in range(0, len(tourn_fit)):
        if fit == tourn_fit[i]:  # if fitness is equal, 50/50 chance of being better
            if r[i] >= 0.5:
                s += 1
        elif fit < tourn_fit[i]:
            s += 1

    return s


# fitness_function, mutation, and constraints are pointer functions to appropriate functions
def evolutionary_programming(init_gen, init_sigma, bounds, fitness_function, constraints, mutation,
                            max_iter=200):
    best_fit = []
    mean_fit = []
    mean_sigmas = []
    gen = np.copy(init_gen)
    sigma = np.copy(init_sigma)
    n, c = gen.shape
    tourn_size = int(0.2*n)
    # weld beam
    temp = [[0.245500, 6.196000, 8.273000, 0.245500],
    [0.244438276, 6.2379672340, 8.2885761430, 0.2445661820],
    [0.2442, 6.2231, 8.2915, 0.2443],
    [0.223100, 1.5815, 12.84680, 0.2245],
    [0.24436895, 6.21860635, 8.29147256, 0.24436895],
    [0.24436198, 6.21767407, 8.29163558, 0.24436883]]
    # compress string
    #temp = np.asarray([[0.051689156131, 0.356720026419, 11.288831695483 ],
    #[0.051749, 0.358179, 11.203763],
    #[0.051515, 0.352529, 11.538862],
    #[0.051688, 0.356692, 11.290483 ]])
    #constraints(temp)
    for k in range(0, max_iter):
        '''
        p1 = np.random.choice(range(0, n), n)
        p2 = np.random.choice(range(0, n), n)
        offspring_values = []
        offspring_sigma = []
        for i in range(0, n):
            ch_v, ch_s = reproduction(gen[p1[i]], gen[p2[i]], sigma[p1[i]], sigma[p2[i]])
            for j in range(0, len(ch_v)):
                offspring_values.append(ch_v[j])
                offspring_sigma.append(ch_s[j])
        '''
        offspring_values, offspring_sigma = mutation(gen, sigma)
        # loop over all individuals
        for i in range(0, len(offspring_values)):
            # loop over all columns/variables
            for j in range(0, c):
                if offspring_values[i][j] > bounds[0][j]:
                    offspring_values[i][j] = bounds[0][j]
                    offspring_sigma[i][j] *= .90
                elif offspring_values[i][j] < bounds[1][j]:
                    offspring_values[i][j] = bounds[1][j]
                    offspring_sigma[i][j] *= .90

        parents_offspring_val = np.vstack((gen, offspring_values))
        parents_offspring_sigma = np.vstack((sigma, offspring_sigma))

        fit = fitness_function(parents_offspring_val)
        # constraint results
        con = constraints(parents_offspring_val)
        # first index of each row above ^ returns how many constraints
        # are broken if it is 0, no constraints broken so we get all
        # indices of the good individuals
        good = np.where(con[:,0] == 0)[0].tolist()
        rel_fit = []
        # loop over each good solution
        for i in range(0, len(good)):
            # tournament indices
            tourn = np.random.choice(good[0:i]+good[(i+1):], tourn_size)
            rel_fit.append(calculate_relative_fit(fit[good[i]], fit[tourn]))
        # now sort indices based off relative fitness in decreasing value
        good = np.asarray(good)[np.argsort(-np.asarray(rel_fit))]

        # get indices of bad solutions, aka when number of constraints broken
        # is not equal to 0
        bad = np.where(con[:,0] != 0)[0].tolist()
        # if has constraints broken, we returned the difference by how much
        # it was broken, so here we get those differences
        violated_con = con[bad][:, 1:]
        # now we calculate the z score by using [x-mean(x)]/std(x)
        z_scores = (violated_con-np.mean(violated_con, axis=0))/(np.std(violated_con, axis=0))
        # if an entire column for a constraint is the same then it
        # would result in nan values, so we replace with 0
        z_scores[np.isnan(z_scores)] = 0
        # sum up the z score for each constraint
        scores = np.sum(z_scores, axis=1)
        # individuals with a larger sum indicate a larger difference broken
        bad = np.asarray(bad)[np.argsort(scores)]

        next_gen_val = np.empty(shape=(n, c))
        next_gen_sigma = np.empty(shape=(n, c))
        # there are more good solutions than needed for next generation
        if len(good) >= n:
            next_gen_val = parents_offspring_val[good[0:n]]
            next_gen_sigma = parents_offspring_sigma[good[0:n]]
        elif len(good) == 0:  # no feasible solutions
            next_gen_val = parents_offspring_val[bad[0:n]]
            next_gen_sigma = parents_offspring_sigma[bad[0:n]]
        else:
            # not enough good solutions so we take all good and the
            # rest are the best bad solutions
            diff = n - len(good)
            next_gen_val[0:len(good)] = parents_offspring_val[good]
            next_gen_sigma[0:len(good)] = parents_offspring_sigma[good]
            next_gen_val[len(good):n] = parents_offspring_val[bad[0:diff]]
            next_gen_sigma[len(good):n] = parents_offspring_sigma[bad[0:diff]]
        gen = next_gen_val
        sigma = next_gen_sigma

        fit = fitness_function(gen)
        fit_best = np.min(fit)
        fit_mean = np.mean(fit)
        sigma_means = []
        for i in range(0, sigma.shape[1]):
            sigma_means.append(np.mean(sigma[:, i]))
        mean_sigmas.append(sigma_means)
        best_fit.append(fit_best)
        mean_fit.append(fit_mean)
        msg = "GENERATION {}:\n" \
              "  Best Fit: {}, Mean Fit: {}".format(k, fit_best, fit_mean)
        if len(good) < n:
            msg = "GENERATION {}:\n" \
                  "  No Feasible Solutions in current Generation".format(k)
        print(msg)
    print("BEST INDIVIDUAL ")
    print(gen[0])
    
    plt.figure(1)
    x_range = range(0, max_iter)
    plt.plot(x_range, mean_fit, label="Mean Fitness")
    plt.plot(x_range, best_fit, label="Best Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Value")
    plt.suptitle("Mean and Best Fitness")
    plt.legend()

    
    mean_sigmas = np.asarray(mean_sigmas)
    for i in range(0, sigma.shape[1]):
        plt.figure(2+i)
        x_range = range(0, max_iter)
        plt.plot(x_range, mean_sigmas[:, i])
        plt.xlabel("Generation")
        plt.ylabel("Strategy Parameter Value")
        plt.suptitle("Mean Strategy Parameter {}".format(i+1))
    plt.show()
    

    return (best_fit[max_iter-1])


size = 1000

# pressure vessel
upper_bound = [6.1875, 6.1875, 200, 240]
lower_bound = [0.0625, 0.0625, 10, 10]
c = 4

total_bound = np.asarray(upper_bound)-np.asarray(lower_bound)
bounds = [upper_bound, lower_bound]

fits = []
for i in range(0, 50):

    init_gen = np.empty(shape=(size, c))
    sigmas = np.empty(shape=(size, c))
    for i in range(0, c):
        init_gen[:, i] = np.random.uniform(lower_bound[i], upper_bound[i], size)
        sigmas[:, i] = np.random.uniform(0.01 * total_bound[i], 0.2 * total_bound[i], size)

    best_fit = evolutionary_programming(init_gen=init_gen, init_sigma=sigmas, bounds=bounds,
                                       fitness_function=pressure_vessel, constraints=constraints,
                                       mutation=mutation, max_iter=100)
    fits.append(best_fit)

print("Me
