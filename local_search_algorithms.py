# Joe Shymanski
# Project 1: Adaptation (Local Search Algorithms)
import ga_eval
import genetic_algorithm as ga
import matplotlib.pyplot as plt
import numpy as np
import pprint
import simulated_annealing as sa
import salesman
import time

def test_saf(f, ranges, min_max, function_name, schedule, T_0, n):
    # Print title
    print(min_max, "of", function_name, "using simulated annealing")

    # Start timer
    start_total = time.time()

    # Run simulated annealing
    print(*sa.simulated_annealing_function(f, ranges, min_max, schedule, T_0, n))
    
    # Stop timer, report time elapsed
    end_total = time.time()
    print(round(end_total - start_total, 2), "seconds")
    print()

def test_sas(cities, min_max, schedule, T_0, n):
    # Print title
    print(min_max, "route for Travelling Salesman Problem using simulated annealing")

    # Start timer
    start_total = time.time()

    # Run simulated annealing on TSP
    sas_cost, sas_route = sa.simulated_annealing_salesman(cities, min_max, schedule, T_0, n)
    print(sas_cost)

    # Stop timer, report time elapsed
    end_total = time.time()
    print(round(end_total - start_total, 2), "seconds")
    print()

    # Plot solution route
    x = []
    y = []
    for city in sas_route:
        x.append(city[0])
        y.append(city[1])
    x.append(sas_route[0][0])
    y.append(sas_route[0][1])
    plt.plot(x, y, marker=".", markersize=10)
    plt.title("SA Solution Route Cost: " + str(sas_cost))
    plt.show()

def test_gaf(f, ranges, min_max, function_name, mut_prob, pop_size, num_gen):
    # Print title
    print(min_max, "of", function_name, "using genetic algorithm")

    # Start timer
    start = time.time()

    # Run genetic algorithm
    print(*ga.genetic_algorithm_function(f, ranges, min_max, mut_prob, pop_size, num_gen))

    # Stop timer, report time elapsed
    end = time.time()
    print(round(end - start, 2), "seconds")
    print()

def test_gas(cities, min_max, mut_prob, pop_size, num_gen):
    # Print title
    print(min_max, "route for Travelling Salesman Problem using genetic algorithm")

    # Start timer
    start_total = time.time()

    # Run genetic algorithm on TSP
    gas_cost, gas_route = ga.genetic_algorithm_salesman(cities, min_max, mut_prob, pop_size, num_gen)
    print(gas_cost)

    # Stop timer, report time elapsed
    end_total = time.time()
    print(round(end_total - start_total, 2), "seconds")
    print()

    # Plot solution route
    x = []
    y = []
    for city in gas_route:
        x.append(city[0])
        y.append(city[1])
    x.append(gas_route[0][0])
    y.append(gas_route[0][1])
    plt.plot(x, y, marker=".", markersize=10)
    plt.title("GA Solution Route Cost: " + str(gas_cost))
    plt.show()

if __name__ == "__main__":
    test_saf(ga_eval.sphere, [[-5, 5], [-5, 5]], "Min", "The Sphere Function", sa.quad_schedule, .1, 10000)
    test_gaf(ga_eval.sphere, [[-5, 5], [-5, 5]], "Min", "The Sphere Function", 1, 85, 90)

    test_saf(ga_eval.griew, [[0, 200], [0, 200]], "Min", "Griewank's function", sa.quad_schedule, .1, 10000)
    test_gaf(ga_eval.griew, [[0, 200], [0, 200]], "Min", "Griewank's function", 1, 85, 90)

    test_saf(ga_eval.shekel, [[0, 10], [0, 10]], "Min", "Modified Shekel's Foxholes", sa.quad_schedule, .1, 10000)
    test_gaf(ga_eval.shekel, [[0, 10], [0, 10]], "Min", "Modified Shekel's Foxholes", 1, 85, 90)

    test_saf(ga_eval.micha, [[-10, 10], [-10, 10]], "Min", "Michalewitz's function", sa.quad_schedule, .1, 10000)
    test_gaf(ga_eval.micha, [[-10, 10], [-10, 10]], "Min", "Michalewitz's function", 1, 85, 90)

    test_saf(ga_eval.micha, [[-10, 10], [-10, 10]], "Max", "Michalewitz's function", sa.quad_schedule, .1, 10000)
    test_gaf(ga_eval.micha, [[-10, 10], [-10, 10]], "Max", "Michalewitz's function", 1, 85, 90)

    test_saf(ga_eval.langermann, [[0, 10], [0, 10]], "Min", "Langermann's function", sa.quad_schedule, .1, 10000)
    test_gaf(ga_eval.langermann, [[0, 10], [0, 10]], "Min", "Langermann's function", 1, 85, 90)

    test_saf(ga_eval.langermann, [[0, 10], [0, 10]], "Max", "Langermann's function", sa.quad_schedule, .1, 10000)
    test_gaf(ga_eval.langermann, [[0, 10], [0, 10]], "Max", "Langermann's function", 1, 85, 90)

    test_saf(ga_eval.odd_square, [[-5 * np.pi, 5 * np.pi], [-5 * np.pi, 5 * np.pi]], "Min", "Odd Square Function", sa.quad_schedule, .1, 10000)
    test_gaf(ga_eval.odd_square, [[-5 * np.pi, 5 * np.pi], [-5 * np.pi, 5 * np.pi]], "Min", "Odd Square Function", 1, 85, 90)

    test_saf(ga_eval.odd_square, [[-5 * np.pi, 5 * np.pi], [-5 * np.pi, 5 * np.pi]], "Max", "Odd Square Function", sa.quad_schedule, .1, 10000)
    test_gaf(ga_eval.odd_square, [[-5 * np.pi, 5 * np.pi], [-5 * np.pi, 5 * np.pi]], "Max", "Odd Square Function", 1, 85, 90)

    test_saf(ga_eval.bump, [[0.1, 5], [0.1, 5]], "Max", "The Bump Function", sa.quad_schedule, .1, 10000)
    test_gaf(ga_eval.bump, [[0.1, 5], [0.1, 5]], "Max", "The Bump Function", 1, 85, 90)

    cities = salesman.random_cities()
    test_sas(cities, "Min", sa.quad_schedule, .5, 100000)
    test_gas(cities, "Min", .01, 180, 200)
