from ern_functions import *
from fenics import *
import numpy as np
import sympy as sym
from matplotlib import rc
import matplotlib.pylab as plt
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


## MODEL PROBLEM 1 ###       
mesh = UnitSquareMesh(10,10)
V = FunctionSpace(mesh, 'CR', 1)   
              
# set up BCs on left and right
def left_boundary(x, on_boundary):
    return on_boundary and abs(x[0]) < 1E-14
def right_boundary(x, on_boundary):
    return on_boundary and abs(x[0]-1) < 1E-14
B1 = DirichletBC(V, Constant(0.0), lambda x, on_boundary : left_boundary(x, on_boundary))
B2 = DirichletBC(V, Constant(1.0), lambda x, on_boundary : right_boundary(x, on_boundary))
bcs = [B1, B2]
exact_solution = Expression('pow((pow(2,m+1) - 1)*x[0] + 1, (1/(m+1))) - 1', m = 2, degree=4)

q = lambda u: (1+u)**2
dqdu = lambda u, u_k: 2*(1+u_k)*(u-u_k)

u1, error_estimators1, Jup1 = solve_2D_flux_PDE(q, Constant(0.0), V, 2, bcs, 
                                        dqdu = dqdu,
                                        exact_solution = exact_solution,
                                        solver = 'Newton',
                                        gamma_lin = 0,
                                        maxIter = 12) # will run until maxIter = 25 is reached
pr = range(10)
plt.figure(1, figsize=(12,10))
plt.semilogy(error_estimators1[pr,0],  'r^-', linewidth=2, markersize=10, label='$\eta_{disc}$')
plt.semilogy(error_estimators1[pr,1],  'b^-', linewidth=2, markersize=10, label='$\eta_{lin}$')
plt.semilogy(error_estimators1[pr,2], 'm^-', linewidth=2, markersize=10, label='$\eta_{quad}$')
#plt.semilogy(error_estimators1[pr,3], label='$\eta_{osc}$')
#plt.semilogy(error_estimators1[pr,4], label='$\eta_{NC}$')
plt.semilogy(error_estimators1[pr,4:6].sum(axis=1),  'g^-', linewidth=2, markersize=10, label='$\eta$')
plt.semilogy(Jup1[0:(pr[-1]+1)],  'k^-', linewidth=2, markersize=10, label='$J_u^{up}$')
plt.semilogy(error_estimators1[pr,0]*0.1,  'r-', linewidth=1.5, markersize=10, label='$\gamma_{lin}\eta_{disc}$')
plt.xlabel('Iterations', fontsize=40)
plt.ylabel('Dual error', fontsize=40)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(loc=0, fontsize=30)

## MODEL PROBLEM 2 ###
#mesh = UnitSquareMesh(10,10)
#V = FunctionSpace(mesh, 'CR', 1)
#x, y= sym.symbols('x[0] x[1]')
#uExSym = 9./10.*((0.5)**(10./9.) - ((x - 0.5)**2 + (y - 0.5)**2)**(5./9.))
#uExpr = Expression(sym.printing.ccode(uExSym), degree=4)
#bcs = DirichletBC(V, uExpr, 'on_boundary')
#u0Sym = uExSym*(1 + (x-0.5)*(y-0.5))
#
#q = lambda u: (inner(grad(u),grad(u)))**4
#dqdu = lambda u, u_k: 8*inner(grad(u_k),grad(u_k))**3*inner(grad(u_k),grad(u - u_k))
#
#u2, error_estimators2, Jup2 = solve_2D_flux_PDE(q, Constant(2.0), V, 10, bcs, 
#                                        dqdu = dqdu,
#                                        u0 = Expression(sym.printing.ccode(u0Sym),degree=4), 
#                                        exact_solution = uExpr,
#                                        solver = 'Newton',
#                                        gamma_lin = 0,
#                                        maxIter = 12)
#pr = range(11)                                     
#plt.figure(2, figsize=(12,10))
#plt.semilogy(error_estimators2[pr,0], 'r^-', linewidth=2, markersize=10, label='$\eta_{disc}$')
#plt.semilogy(error_estimators2[pr,1], 'b^-', linewidth=2, markersize=10, label='$\eta_{lin}$')
##plt.semilogy(error_estimators2[pr,2], 'm^-', linewidth=2, markersize=10, label='$\eta_{quad}$')
##plt.semilogy(error_estimators2[pr,3], label='$\eta_{osc}$')
##plt.semilogy(error_estimators2[pr,4], label='$\eta_{NC}$')
#plt.semilogy(error_estimators2[pr,0:4].sum(axis=1),  'g^-', linewidth=2, markersize=10, label='$\eta$')
#plt.semilogy(Jup2[0:(pr[-1]+1)],  'k^-', linewidth=2, markersize=10, label='$J^{up}$')
#plt.semilogy(error_estimators2[pr,0]*0.1,  'g-', linewidth=1.5, markersize=10, label='$\gamma_{lin}\eta_{disc}$')
#plt.xlabel('Iterations', fontsize=40)
#plt.ylabel('Dual error', fontsize=40)
#plt.xticks(fontsize=25)
#plt.yticks(fontsize=25)
#plt.legend(loc=0, fontsize=30)
