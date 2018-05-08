from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from getPDEConvData import *

m = 2;
def q(u):
    return (1+u)**m
    
def ue(x):
    return ((2**(m+1)-1)*x + 1)**(1/(m+1)) - 1 

# Define variational problem for Picard iteration
u = TrialFunction(V)
v = TestFunction(V)
u_k = interpolate(Constant(0.0), V)  # previous (known) u
a = inner(q(u_k)*grad(u), grad(v))*dx
f = Constant(0.0)
L = f*v*dx

# set up BCs on left and right
BCtol = 1E-14
def left_boundary(x, on_boundary):
    return on_boundary and abs(x[0]) < BCtol
    
def right_boundary(x, on_boundary):
    return on_boundary and abs(x[0]-1) < BCtol
    
B1 = DirichletBC(V, Constant(0.0), left_boundary) # u(0) = 0
B2 = DirichletBC(V, Constant(1.0), right_boundary) # u(1) = 1
bcs = [B1, B2]

[uDiff, exactError] = getPDEConvData(a, f, bcs, ue, 100, 1, dispOutput = True)