from ern_functions import *
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym



#def q(u):
#    #return (1+u)**2 
#    return (inner(grad(u),grad(u)))**4
#def dqdu(u):
#    #return 2*(1+u)
#    return None
#def dqdgu(u):
#    #return None
#    return 8*inner(grad(u),grad(u))**3*grad(u)
#def sigma(u):
#    return q(u)*grad(u)
#def sigma_Picard(u, u_k):
#    return q(u_k)*grad(u)
#def sigma_Newton(u, u_k):
#    SN = q(u_k)*grad(u)
#    if dqdu(u) != None: SN += (u - u_k)*dqdu(u_k)*grad(u_k)
#    if dqdgu(u) != None: SN += grad(u - u_k)*inner(dqdgu(u_k), grad(u_k))
#    return SN

### MODEL PROBLEM 2 ###
#mesh = UnitSquareMesh(10,10)
#V = FunctionSpace(mesh, 'CR', 1)
#x, y= sym.symbols('x[0] x[1]')
#uExSym = 9./10.*((0.5)**(10./9.) - ((x - 0.5)**2 + (y - 0.5)**2)**(5./9.))
#uExpr = Expression(sym.printing.ccode(uExSym), degree=4)
#bcs = DirichletBC(V, uExpr, 'on_boundary')
#u0Sym = uExSym*(1 + (x-0.5)*(y-0.5))
#q = lambda u: (inner(grad(u),grad(u)))**4
#dqdu = lambda u, u_k: 8*inner(grad(u_k),grad(u_k))**3*inner(grad(u_k),grad(u - u_k))
#u, error_estimators = solve_2D_flux_PDE(q, Constant(2.0), V, 10, bcs, 
#                                        dqdu = dqdu,
#                                        u0 = Expression(sym.printing.ccode(u0Sym),degree=4), 
#                                        exact_solution = uExpr,
#                                        solver = 'Newton')

### MODEL PROBLEM 1 ###       
mesh = UnitSquareMesh(10,10)
V = FunctionSpace(mesh, 'CR', 1)                 
# set up BCs on left and right
# lambda functions ensure the boundary methods take two variables
def left_boundary(x, on_boundary):
    return on_boundary and abs(x[0]) < 1E-14
def right_boundary(x, on_boundary):
    return on_boundary and abs(x[0]-1) < 1E-14
B1 = DirichletBC(V, Constant(0.0), lambda x, on_boundary : left_boundary(x, on_boundary))
B2 = DirichletBC(V, Constant(1.0), lambda x, on_boundary : right_boundary(x, on_boundary))
bcs = [B1, B2]
exact_solution = Expression('pow((pow(2,m+1) - 1)*x[0] + 1, (1/(m+1))) - 1', m = 2, degree=4)

q = lambda u: (1+u)**2
dqdu = lambda u, u_k: 2*(1+u_k)(u-u_k)

u, error_estimators = solve_2D_flux_PDE(q, Constant(0.0), V, 2, bcs, 
                                        dqdu = dqdu,
                                        exact_solution = exact_solution,
                                        solver = 'Picard')
                   
#%% PLOTTING
                     
plt.figure()
plt.semilogy(error_estimators[:,0], label='$\eta_{disc}$')
plt.semilogy(error_estimators[:,1], label='$\eta_{lin}$')
#plt.semilogy(error_estimators[:,2], label='$\eta_{quad}$')
#plt.semilogy(error_estimators[:,3], label='$\eta_{osc}$')
#plt.semilogy(error_estimators[:,4], label='$\eta_{NC}$')
plt.semilogy(error_estimators[:,0:4].sum(axis=1), label='$\eta$')
#plt.semilogy(exactErrArray, label='$||u_h - u_e||_{H1}$')
#plt.semilogy(JupArray, label='$J^{up}$')
plt.xlabel('Iterations')
plt.ylabel('$\eta$')
plt.legend()
