# Joe Shymanski
# Project 1: Adaptation (Simulated Annealing)
import math
import numpy as np
import pprint
import random
import salesman
import time

def lin_schedule(k, T_0, n):
    return T_0*((n-k)/n)

def quad_schedule(k, T_0, n):
    return T_0*((n-k)/n)**2

def trig_schedule(k, T_0, n):
    return .5*T_0*(1+math.cos(k*math.pi/n))

def simulated_annealing_function(f, ranges, min_max, schedule, T_0, n):
    # Select random initial point
    curr_point = np.array([])
    for i in range(len(ranges)):
        rand = random.uniform(*ranges[i])
        curr_point = np.append(curr_point, rand)
    curr_val = f(curr_point)

    # Loop according to annealing schedule
    for k in range(n):
        # Change temperature according to annealing schedule
        T = schedule(k, T_0, n)

        # Randomly pick next point within a neighborhood from current point
        next_point = np.array([])
        for i in range(len(ranges)):
            maxi = max(ranges[i])
            mini = min(ranges[i])
            std = (maxi - mini)/5
            rand = random.gauss(curr_point[i], std)
            # Ensure next point is within range
            while rand < mini or rand > maxi:
                rand = random.gauss(curr_point[i], std)
            next_point = np.append(next_point, rand)
        next_val = f(next_point)

        # Check to see if next point is improvement
        delta_E = curr_val - next_val if min_max == "Min" else next_val - curr_val
        if delta_E > 0:
            curr_point = next_point
            curr_val = next_val
        # If not, choose it with probability according to Boltzmann distribution
        else:
            probability = math.exp(delta_E / T)
            if random.random() < probability:
                curr_point = next_point
                curr_val = next_val

    return (curr_point, curr_val)

def simulated_annealing_salesman(cities, min_max, schedule, T_0, n):
    # Generate initial route cost
    curr_route = cities
    curr_cost = salesman.route_cost(curr_route)

    # Loop according to annealing schedule
    for k in range(n):
        # Change temperature according to annealing schedule
        T = schedule(k, T_0, n)

        # Perform action on a random pair of cities
        next_route = curr_route[:]
        l = len(next_route)
        c1 = random.randrange(0, l)
        c2 = random.randrange(0, l)
        # Ensure next city is different
        while c1 == c2:
            c2 = random.randrange(0, l)
        if random.random() < .5:
            # Swap the cities
            tmp = next_route[c1]
            next_route[c1] = next_route[c2]
            next_route[c2] = tmp
        else:
            # Insert city2 behind city1
            city2 = next_route.pop(c2)
            next_route.insert((c1 + 1) % l, city2)
        next_cost = salesman.route_cost(next_route)

        # Check to see if next route is improvement
        delta_E = curr_cost - next_cost if min_max == "Min" else next_cost - curr_cost
        if delta_E > 0:
            curr_route = next_route
            curr_cost = next_cost
        # If not, choose it with probability according to Boltzmann distribution
        else:
            probability = math.exp(delta_E / T)
            if random.random() < probability:
                curr_route = next_route
                curr_cost = next_cost

    return (curr_cost, curr_route)
