from fenics import *
import matplotlib.pyplot as plt
import numpy as np

def left_boundary(x, on_boundary):
        return on_boundary and abs(x[0]) < 1E-14
def right_boundary(x, on_boundary):
        return on_boundary and abs(x[0]-1) < 1E-14
def q(u):
        return (1+u)**2
def dqdu(u):
        return 2*(1+u)
        
N = 10
p = 1

mesh = UnitSquareMesh(N,N)
V = FunctionSpace(mesh, 'CR', p)

# set up BCs on left and right
# lambda functions ensure the boundary methods take two variables
B1 = DirichletBC(V, Constant(0.0), lambda x, on_boundary : left_boundary(x, on_boundary)) # u(0) = 0
B2 = DirichletBC(V, Constant(1.0), lambda x, on_boundary : right_boundary(x, on_boundary)) # u(1) = 1

# construct exact solution in C format
uExpr = Expression('pow((pow(2,m+1) - 1)*x[0] + 1,(1/(m+1))) - 1', m = 2, degree=4)

iterTol = 1.0e-5; maxIter = 25; dispOutput = True

V = V
u = TrialFunction(V)
v = TestFunction(V)
u_k = interpolate(Constant(0.0), V) # previous (known) u
a = inner(q(u_k)*grad(u), grad(v))*dx
f = Constant(0.0)
L = f*v*dx

bcs = [B1, B2] 

X = SpatialCoordinate(mesh)
CR0 = FunctionSpace(mesh, 'CR', 0)
CR1 = FunctionSpace(mesh, 'CR', 1)
f_h = interpolate(f, CR0)
x_ = interpolate(Expression('x[0]', degree=1), CR1)
d = 2 # dimension of the space
f_hvec = f_h/d * (x - x_K) #TODO: calculate barycenters x_K

u = Function(V)     # new unknown function
itErr = 1.0           # error measure ||u-u_k||
iterDiffArray = []
exactErrArray = []
iter = 0

# Begin Picard iterations
while itErr > iterTol and iter < maxIter:
    iter += 1
    
    solve(a == L, u, bcs)
    
    # calculate iterate difference and exact error in L2 norm
    itErr = errornorm(u_k, u, 'H1')
    exErr = errornorm(uExpr, u, 'H1')
    
    iterDiffArray.append(itErr) # fill arrays with error data
    exactErrArray.append(exErr)
    
    if dispOutput:
        print('k = ' + str(iter) + ' | u-diff =  ' + str(itErr) + ', exact error = ' + str(exErr))
    u_k.assign(u) # update for next ite