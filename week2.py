from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from getPDEConvData import *

#%%************** Problem 1 **************

m = 2;
def q(u):
    return (1+u)**m
    
#def ue(x):
#    return ((2**(m+1)-1)*x + 1)**(1/(m+1)) - 1 
#    
# set up BCs on left and right
BCtol = 1E-14
def left_boundary(x, on_boundary):
    return on_boundary and abs(x[0]) < BCtol
    
def right_boundary(x, on_boundary):
    return on_boundary and abs(x[0]-1) < BCtol
N = 100
d = 1
if d == 1:
    mesh = UnitIntervalMesh(N)
elif d == 2:
    mesh = UnitSquareMesh(N,N)
else:
    #experiments in higher dimensions are too expensive
    raise NotImplementedError


V = FunctionSpace(mesh, 'CG', p)

# construct exact answer in C format
uExpr = Expression('pow((pow(2,m+1) - 1)*x[0] + 1,(1/(m+1))) - 1', m = 2.0, degree=3)
# interpolate over function space
ue = interpolate(uExpr, V)

# Define variational problem for Picard iteration
u = TrialFunction(V)
v = TestFunction(V)
u_k = interpolate(Constant(0.0), V)  # previous (known) u
a = inner(q(u_k)*grad(u), grad(v))*dx
f = Constant(0.0)
L = f*v*dx


B1 = DirichletBC(V, Constant(0.0), left_boundary) # u(0) = 0
B2 = DirichletBC(V, Constant(1.0), right_boundary) # u(1) = 1
bcs = [B1, B2]


[u_k, uDiff, exactError] = getPDEConvData(a, L, u_k, ue, V, bcs, dispOutput = True)
plt.figure(1)
plt.plot(uDiff, label='Iterate difference')
plt.plot(exactError, label='Exact error')
plt.legend()
V = FunctionSpace(mesh, 'CG', 2)



# Define variational problem for Picard iteration
#u = TrialFunction(V)
#v = TestFunction(V)
#u_k = interpolate(Constant(0.0), V)  # previous (known) u
#a = inner(q(u_k)*grad(u), grad(v))*dx
#f = Constant(0.0)
#L = f*v*dx
#
#B1 = DirichletBC(V, Constant(0.0), left_boundary) # u(0) = 0
#B2 = DirichletBC(V, Constant(1.0), right_boundary) # u(1) = 1
#bcs = [B1, B2]
#
#[uDiff2, exactError2] = getPDEConvData(a, f, u_k, ue, mesh, 1, V, 100, 2, bcs, dispOutput = True)
#plt.figure(2)
#plt.plot(uDiff2, label='Iterate difference') 
#plt.plot(exactError2, label='Exact error')
#plt.legend()



