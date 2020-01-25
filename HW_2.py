#!/usr/bin/env python3.7

import numpy as np
import gurobipy as gp
from gurobipy import GRB

x = np.array([1, 0, 0, 1, 2, 2, 1, 2, 0, 1, 3, 0]).reshape((4,3))
y = np.array([-1, -1, 1, 1])
A = np.zeros((x.shape))
for i in range(x.shape[0]):
    A[i] = [y[i], y[i]*x[i][-2], y[i]*x[i][-1]]
Q = np.eye(3)
Q[0][0] = 0
p = np.zeros(Q.shape[0])
c = np.ones(x.shape[0])

# Create a new model
m = gp.Model("qp")

# Create variables
b = m.addVar(name='b', lb=-10.0)
w0 = m.addVar(name='w0', lb=-10.0)
w1 = m.addVar(name='w1', lb=-10.0)

u = np.array([b, w0, w1])
# Set objective: x
obj = 1/2 * (u.dot(Q)).dot(u) + p.dot(u)

m.setObjective(obj, GRB.MINIMIZE)

# Add constraint: A dot u >= 1
for i in range(x.shape[0]):
    m.addConstr(A[i].dot(u) >= c[i], "c" + str(i))

m.optimize()

for v in m.getVars():
    print('%s %g' % (v.varName, v.x))

print('Obj: %g' % obj.getValue())

print("\nThe output for the u vector is")
for i, v in enumerate(m.getVars()):
    print('%s %g' % ("u"+str(i), v.x))

print("\nQ = ", Q, "\n")

print("p = ", p, "\n")

print("A = ", A, "\n")

print("c = ", c, "\n")