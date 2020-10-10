
import sys
import math
from itertools import combinations
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random
from matplotlib import pyplot as plt



# Generate sets of points that will be split up later
random.seed(1)
# when converting the random tuples to a set, it only keep unique tuples
points_co = set([(random.randint(0, 4), random.randint(0, 12)) for i in range(100)])
points_co = list(points_co)[:60]

unique_points = np.array(list(points_co))
x = unique_points[:, 0]
y = unique_points[:, 1]


# Logic for splitting
def split_logic(point, ax_break, diff):
    # logic can be changed whenever required. Currently, this logic coincidentally leads to two salesmen picking up one set on the border due to the <= and >=
    if point[1] >= ax_break and point[1] < ax_break + diff:
        return True
    else:
        return False


n_salesmen = 3
point_dict = {i: unique_points[i] for i in range(len(unique_points))}
accessible_sets_mask = {i: np.zeros((len(unique_points), 1)) for i in range(n_salesmen)}

private_sets = {}
shared_sets = {}

y_break = 0
# Split the set up into 3 accessible sets based on y coordinate
for i in range(n_salesmen):
    temp_priv_list = {}
    temp_shared_list = {}
    for index in range(len(point_dict.values())):

        if split_logic(point_dict[index], y_break, 3):
            temp_priv_list[index] = point_dict[index]
            accessible_sets_mask[i][index] = 1

        if split_logic(point_dict[index], y_break + 3, 2):
            temp_shared_list[index] = point_dict[index]
            accessible_sets_mask[i][index] = 1
            accessible_sets_mask[i + 1][index] = 1
    y_break += 5
    private_sets[i] = temp_priv_list
    shared_sets[i] = temp_shared_list

shared_sets.pop(n_salesmen - 1)
n = len(accessible_sets_mask[0])
m = n_salesmen
salesmen = range(1, m + 1)

# Build Python Sets (these allow mathematical set operations)

private_sets = {k + 1: {key for key in private_sets[k].keys()} for k in private_sets.keys()}
private_sets[0] = set()
private_sets[n_salesmen + 1] = set()
shared_sets = {k + 1: {key for key in shared_sets[k].keys()} for k in shared_sets.keys()}
shared_sets[0] = set()
shared_sets[n_salesmen] = set()

accessible_sets = {k + 1: shared_sets[k].union(private_sets[k + 1].union(shared_sets[k + 1])) for k in
                   range(n_salesmen)}
all_points = {point for point in point_dict.keys()}


# Calculating the Distance Matrix
dist = {(i, j):
            math.sqrt(sum((points_co[i][k] - points_co[j][k]) ** 2 for k in range(2)))
        for i in all_points for j in all_points}


model = gp.Model()

# Variables

x = {}
for i in all_points:
    for j in all_points:
        if i != j:
            for k in salesmen:
                x[i, j, k] = model.addVar(vtype=GRB.BINARY, obj=dist[i, j],
                                          name="x_" + str(i) + "_" + str(j) + "_" + str(k))

u = {}
for i in all_points:
    for k in salesmen:
        u[i, k] = model.addVar(vtype=GRB.INTEGER, name="u_" + str(i) + "_" + str(k))


# Modified Constraints

# Not allowed to travel to or from cities not present in the accessible set
for k in salesmen:
    for i in accessible_sets[k]:
        for j in all_points.difference(accessible_sets[k]):
            model.addConstr(x[i, j, k] == 0)
            model.addConstr(x[j, i, k] == 0)


# Not allowed to travel between cities not present in the accessible set
for k in salesmen:
    for i in all_points.difference(accessible_sets[k]):
        for j in all_points.difference(accessible_sets[k]):
            if i != j:
                model.addConstr(x[i, j, k] == 0)
                model.addConstr(x[j, i, k] == 0)


# each city needs to be visited exactly once
for i in all_points:
    model.addConstr(gp.quicksum(gp.quicksum(x[j, i, k] for j in all_points if i != j) for k in salesmen) == 1)


# each city which is visited by a salesmen must be travelled away from by the same salesman
for k in salesmen:
    for i in accessible_sets[k]:
        model.addConstr(gp.quicksum(x[i, j, k] for j in accessible_sets[k] if i != j) ==
                        gp.quicksum(x[j, i, k] for j in accessible_sets[k] if i != j))


# Subtour Elimination Constraints 
for k in salesmen:
    for i in all_points.difference(set([list(private_sets[k])[0]])):
        for j in all_points.difference(set([list(private_sets[k])[0]])):
            if i != j:
                model.addConstr(
                    u[i, k] - u[j, k] + len(accessible_sets[k]) * x[
                        i, j, k] <=
                    len(accessible_sets[k]) - 1)


# Model Settings

model.write("ctsp.lp")
model._vars = vars


model.setParam("TimeLimit", 120)
model.optimize()

try:
    vals = model.getAttr('x', x)
    selected = gp.tuplelist((i, j, k) for i, j, k in vals.keys() if vals[i, j, k] > 0.5)

    print('')
    # print('Optimal tour: %s' % str(tour))
    print('Optimal cost: %g' % model.objVal)
    print('')

    # Print results
    for i in all_points:
        for j in all_points:
            if j != i:
                for k in salesmen:
                    if vals[i, j, k] > 0.5:
                        print('The Salesman ' + str(k) + ' travels directly from city ' + str(i) + ' to city ' + str(j))

    
    # Plot cities

    for key, set in private_sets.items():
        for point in set:
            fig = plt.plot(points_co[point][1], points_co[point][0], "D", color="purple", markersize=5)
            plt.annotate(point, (points_co[point][1], point_dict[point][0]), fontsize=12)
    for key, set in shared_sets.items():
        for point in set:
            fig = plt.plot(points_co[point][1], points_co[point][0], "D", color="green", markersize=5)
            plt.annotate(point, (point_dict[point][1], point_dict[point][0]), fontsize=12)


    # Plot arcs

    for i in all_points:
        for j in all_points:
            if j != i:
                for k in salesmen:
                    if vals[i, j, k] > 0.5:
                        if k == 1:
                            fig = plt.plot([points_co[i][1], points_co[j][1]], [points_co[i][0], points_co[j][0]],
                                           color="black",
                                           alpha=vals[i, j, k], linewidth=2)
                        elif k == 2:
                            fig = plt.plot([points_co[i][1], points_co[j][1]], [points_co[i][0], points_co[j][0]],
                                           color="blue",
                                           alpha=vals[i, j, k], linewidth=2)
                        elif k == 3:
                            fig = plt.plot([points_co[i][1], points_co[j][1]], [points_co[i][0], points_co[j][0]],
                                           color="red",
                                           alpha=vals[i, j, k], linewidth=2)
                        else:
                            pass

    plt.savefig('ctsp.png')
    plt.show()
except:
    print("No integer solution found")
