# Joe Shymanski
# Project 1: Adaptation (Genetic Algorithm)
import ga_util
import math
import numpy as np
import pprint
import random
import salesman
import time

def bin_point_to_dec_point(point, ranges):
    point_dec = np.array([])
    for i in range(len(point)):
        maxi = max(ranges[i])
        mini = min(ranges[i])
        # Scale float to fit range of function
        val = ga_util.bitstr2float(point[i])*(maxi - mini) + mini
        point_dec = np.append(point_dec, val)
    return point_dec

def reproduce_point(parent1, parent2):
    child = []
    for i in range(len(parent1)):
        n = len(parent1[i])
        c = random.randrange(1, n)
        # Crossover at index c
        child.append(parent1[i][:c] + parent2[i][c:])
    return child

def reproduce_route(parent1, parent2):
    # Make copy of parent1
    child = parent1[:]
    n = len(child)
    # Choose crossover point c
    c = random.randrange(1, n)
    # Loop over first c cities
    for index1 in range(c):
        # city1 and city2 are at same index in different parents
        city1 = child[index1]
        city2 = parent2[index1]
        if city1 != city2:
            # Swap city1 and city2 within parent1 copy
            index2 = child.index(city2)
            child[index1] = city2
            child[index2] = city1
    return child

def mutate_point(child):
    for i in range(len(child)):
        n = len(child[i])
        c = random.randrange(0, n)
        # Flip bit at index c
        child[i] = child[i][:c] + str(int(child[i][c]) ^ 1) + child[i][c + 1:]

def mutate_route(child):
    n = len(child)
    # Pick two cities to swap
    c1 = random.randrange(0, n)
    c2 = random.randrange(0, n)
    # Ensure cities are different
    while c1 == c2:
        c2 = random.randrange(0, n)
    # Swap the cities
    tmp = child[c1]
    child[c1] = child[c2]
    child[c2] = tmp

def genetic_algorithm_function(f, ranges, min_max, mut_prob, pop_size, num_gen):
    # Build initial, random population
    population = []
    for _ in range(pop_size):
        point = []
        for i in range(len(ranges)):
            bitstr = ""
            for _ in range(52):
                bitstr += str(random.randint(0, 1))
            point.append(bitstr)
        population.append(point)

    # Loop through generations
    for _ in range(num_gen):
        # Calculate weights from point heights
        weights = []
        for point in population:
            point_dec = bin_point_to_dec_point(point, ranges)
            fitness = f(point_dec)
            weights.append(fitness)
        # Flip weights if finding minimum
        if min_max == "Min":
            weights = [-x for x in weights]
        # Ensure weights total more than 0
        min_weight = min(weights)
        weights = [x - min_weight + 1 for x in weights]

        # Choose parents from weights, reproduce, and mutate
        population2 = []
        for __ in range(len(population)):
            # Time consuming
            parent1, parent2 = random.choices(population, weights=weights, k=2)
            child = reproduce_point(parent1, parent2)
            if random.random() < mut_prob: mutate_point(child)
            population2.append(child)
        population = population2

    # Return best point
    best_point = None
    best_fitness = math.inf if min_max == "Min" else -math.inf
    for point in population:
        point_dec = bin_point_to_dec_point(point, ranges)
        fitness = f(point_dec)
        if min_max == "Min" and fitness < best_fitness or min_max == "Max" and fitness > best_fitness:
            best_point = point_dec
            best_fitness = fitness
    return (best_point, best_fitness)

def genetic_algorithm_salesman(cities, min_max, mut_prob, pop_size, num_gen):
    # Build initial, random population
    population = []
    for _ in range(pop_size):
        route = cities[:]
        random.shuffle(route)
        population.append(route)

    # Loop through generations
    for _ in range(num_gen):
        # Calculate weights from point heights
        weights = []
        for route in population:
            fitness = salesman.route_cost(route)
            weights.append(fitness)
        # Flip weights if finding minimum
        if min_max == "Min":
            weights = [-x for x in weights]
        # Ensure weights total more than 0
        min_weight = min(weights)
        weights = [x - min_weight + 1 for x in weights]

        # Choose parents from weights, reproduce, and mutate
        population2 = []
        for __ in range(len(population)):
            parent1, parent2 = random.choices(population, weights=weights, k=2)
            child = reproduce_route(parent1, parent2)
            if random.random() < mut_prob: 
                mutate_route(child)
            population2.append(child)
        population = population2

    # Return best point
    best_route = None
    best_fitness = math.inf if min_max == "Min" else -math.inf
    for route in population:
        fitness = salesman.route_cost(route)
        if min_max == "Min" and fitness < best_fitness or min_max == "Max" and fitness > best_fitness:
            best_route = route
            best_fitness = fitness
    return (best_fitness, best_route)
