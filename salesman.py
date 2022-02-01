import math
import random

def random_cities(n=30):
    l = []
    for _ in range(n):
        l.append((round(random.random(),4),round(random.random(),4)))
    return l

cities = [
    (0.8032, 0.3889), (0.4106, 0.6614), (0.7381, 0.9347), (0.3419, 0.4497), (0.7731, 0.1003),
    (0.6144, 0.5277), (0.5476, 0.1709), (0.9947, 0.0195), (0.9466, 0.7510), (0.3386, 0.0721),
    (0.6602, 0.7499), (0.3465, 0.7377), (0.1170, 0.3852), (0.5075, 0.6245), (0.2351, 0.2451),
    (0.2221, 0.3332), (0.2415, 0.1832), (0.2462, 0.9953), (0.3537, 0.3027), (0.6727, 0.8953),
    (0.5095, 0.4163), (0.5836, 0.6556), (0.1333, 0.0102), (0.2176, 0.5716), (0.9814, 0.3870),
    (0.5864, 0.5520), (0.4856, 0.1720), (0.1029, 0.8778), (0.5735, 0.8530), (0.3865, 0.6119)
]

def route_cost(route):
    cost = 0
    curr_city = route[0]
    for next_city in route[1:]:
        cost += math.dist(curr_city, next_city)
        curr_city = next_city
    next_city = route[0]
    cost += math.dist(curr_city, next_city)
    return cost
