from fenics import *
import matplotlib.pyplot as plt
import numpy as np
from fenics_adjoint import *

N=100;
mesh = UnitSquareMesh(N,N)

x = SpatialCoordinate(mesh)
ue = 10*sin(4*pi*x[0])*sin(pi*x[1])

alpha = 1e-10

p=1;
V = FunctionSpace(mesh, 'CG', p)
W = FunctionSpace(mesh, 'CG', p)

# initial guess for control
m = interpolate(Expression("x[0]", degree=1), V)

u = TrialFunction(V)
v = TestFunction(V)
# solve state equation 
a = inner(grad(u), grad(v))*dx
L = m*v*dx
bc = DirichletBC(V, 0., 'on_boundary')
# dolfin_adjoint automatically saves iterates for use in control problem
u = Function(V, name='State')
solve(a == L, u, bc)

ud = ue#interpolate(ue, V)

J_form = (0.5*inner(u-ud, u-ud))*dx + alpha/2*m**2*dx # + lmbd*(inner(grad(u), grad(v)) - m*v)*dx
J_form_u = (0.5*inner(u-ud, u-ud))*dx
J_form_m = m**2*dx
J = assemble(J_form)
control = Control(m)
rf = ReducedFunctional(J, control)
dJdm = rf.derivative()

#Riesz representative
RdJ_ = TrialFunction(V)
RdJ = Function(V)
a_R = RdJ_*v*dx
L_R = dJdm*v*dx

Jk = [assemble(J_form)]
mDiff = []
RdJk = []
iter = 0

iterTol = 1e-05; maxIter = 25; dispOutput = True
print(str(assemble(J_form_m)))
while Jk[-1] > iterTol and iter < maxIter:
    iter += 1
    
    # solve state eqn
    solve(a == L, u, bc)
    dJdm = rf.derivative()
    
    # compute Riesz rep. (step-direction)
    solve(a_R == L_R, RdJ)
    # step
    m.assign(m - 100000*RdJ, annotate=False)
    
    # solve state eqn and evaluate objective
    solve(a == L, u, bc)
    Jk.append(assemble(J_form))
    
    if dispOutput:
        print ('k = ' + str(iter) + ' | J = ' + str(assemble(J_form_u)) + ' | norm(m) = ' 
            + str(norm(m, 'H1')) + ' | norm(R(dJ)) = ' + str(norm(dJdm, 'H1')))
        #print ('k = ' + str(iter) + ' | J = ' + str(Jk[-1]) + ' | norm(m) = ' 
        #    + str(norm(m, 'H1')) + ' | norm(R(dJ)) = ' + str(norm(dJdm, 'H1')))