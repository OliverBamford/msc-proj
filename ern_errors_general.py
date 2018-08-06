from ern_functions import *
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

def left_boundary(x, on_boundary):
    return on_boundary and abs(x[0]) < 1E-14
def right_boundary(x, on_boundary):
    return on_boundary and abs(x[0]-1) < 1E-14
def q(u):
    #return (1+u)**2 
    return (inner(grad(u),grad(u)))**4
def dqdu(u):
    #return 2*(1+u)
    return None
def dqdgu(u):
    #return None
    return 8*inner(grad(u),grad(u))**3*grad(u)
def sigma(u):
    return q(u)*grad(u)
def sigma_Picard(u, u_k):
    return q(u_k)*grad(u)
def sigma_Newton(u, u_k):
    SN = q(u_k)*grad(u)
    if dqdu(u) != None: SN += (u - u_k)*dqdu(u_k)*grad(u_k)
    if dqdgu(u) != None: SN += grad(u - u_k)*inner(dqdgu(u_k), grad(u_k))
    return SN

### MESH SETUP ###
N = 15 # mesh fine-ness
l = 1 # order of finite elements
d = 2 # dimension of the space
mesh = UnitSquareMesh(N,N)
n = FacetNormal(mesh)
mesh.init()
V = FunctionSpace(mesh, 'CR', l)

### BOUNDARY CONDITIONS ###
x, y= sym.symbols('x[0] x[1]')
uExSym = 9./10.*((0.5)**(10./9.) - ((x - 0.5)**2 + (y - 0.5)**2)**(5./9.))
uExpr = Expression(sym.printing.ccode(uExSym), degree=4)
u_e = interpolate(uExpr, V)
bcs = DirichletBC(V, uExpr, 'on_boundary')

# set up BCs on left and right
# lambda functions ensure the boundary methods take two variables
#B1 = DirichletBC(V, Constant(0.0), lambda x, on_boundary : left_boundary(x, on_boundary))
#B2 = DirichletBC(V, Constant(1.0), lambda x, on_boundary : right_boundary(x, on_boundary))
#bcs = [B1, B2]
#uExpr = Expression('pow((pow(2,m+1) - 1)*x[0] + 1, (1/(m+1))) - 1', m = 2, degree=4)
#u_e = interpolate(uExpr, V)
### INITIAL GUESS ###
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(2.0)
DG0 = FunctionSpace(mesh, 'DG', 0)
f_h = interpolate(f, DG0)
#u_k = TrialFunction(V)
#a0 = inner(grad(u_k), grad(v))*dx
#L0 = f_h*v*dx
#u_k = Function(V)
#solve(a0 == L0, u_k, bcs)
#u_k = interpolate(Constant(0.0), V) # previous (known) u
u0Sym = uExSym*(1 + (x-0.3)*(y-0.3))
u_k = interpolate(Expression(sym.printing.ccode(u0Sym), degree=4), V)

### GET REF SOLUTION ###
#u_ref = u_k
#F = (inner((inner(grad(u_ref),grad(u_ref)))**4*grad(u_ref), grad(v)) - f_h*v)*dx
#solve(F==0, u_ref, bcs)

### PROBLEM SET UP ###
p = 10
qu = p/float(p-1)
sigma_lin = lambda u, u_k: sigma_Newton(u, u_k)
F = (inner(sigma_lin(u, u_k), grad(v)) - f_h*v)*dx
a = lhs(F)
L = rhs(F)

f_hvec = get_fvec(f_h, mesh)

u = Function(V)
eta_lin = 1.
eta_disc = 0.
gamma_lin = 0.0000001
maxIter = 25; dispOutput = True
iterDiffArray = [0]
exactErrArray = [0]
iter = 0
error_estimators = np.array([[0.,0.,0.,0.,0.]])
JupArray = [0]
# Begin Picard iterations
while eta_lin > gamma_lin*eta_disc and iter < maxIter:
    iter += 1
    
    solve(a == L, u, bcs, solver_parameters={"linear_solver": "lu"})

    # calculate iterate difference and exact error in L2 norm
    itErr = errornorm(u_k, u, 'H1')
    exErr = errornorm(uExpr, u, 'H1')
    iterDiffArray.append(itErr) # fill arrays with error data
    exactErrArray.append(exErr)
    
    eta_disc, eta_lin, eta_quad, eta_osc, eta_NC = get_estimators(V, f, f_h, f_hvec, sigma, sigma_lin, u, u_k, mesh, p)
    error_estimators = np.concatenate((error_estimators, np.array([[eta_disc, eta_lin, eta_quad, eta_osc, eta_NC]])), axis = 0)
    JupArray.append(assemble((inner(sigma(u_e) - sigma(u), sigma(u_e) - sigma(u)))**(qu/2)*dx)**(1/qu) + error_estimators[-1,4])
    if dispOutput:
        print('k = ' + str(iter) + ' | u-diff =  ' + str(itErr) + 
        ', exact error = ' + str(exErr) + ', eta_lin = ' + str(eta_lin)
        + ' , eta_disc = ' + str(eta_disc))
    u_k.assign(u) # update for next iteration

plt.figure()
plt.semilogy(error_estimators[:,0], label='$\eta_{disc}$')
plt.semilogy(error_estimators[:,1], label='$\eta_{lin}$')
#plt.semilogy(error_estimators[:,2], label='$\eta_{quad}$')
plt.semilogy(error_estimators[:,3], label='$\eta_{osc}$')
plt.semilogy(error_estimators[:,4], label='$\eta_{NC}$')
plt.semilogy(error_estimators[:,0:4].sum(axis=1), label='$\eta$')
plt.semilogy(exactErrArray, label='$||u_h - u_e||_{H1}$')
plt.semilogy(JupArray, label='$J^{up}$')
plt.xlabel('Iterations')
plt.ylabel('$\eta$')
plt.legend()