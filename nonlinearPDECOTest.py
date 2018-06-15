from fenics import *
import matplotlib.pyplot as plt
import numpy as np
from fenics_adjoint import *

#L2-tracking functional with Tykhonov regularitation
#control is the source function of Poisson's eqn
mesh = UnitSquareMesh(100,100)

#target function
x = SpatialCoordinate(mesh)
ue = 10*sin(4*pi*x[0])*sin(pi*x[1])

#Tykhonov regularization
alpha = 1e-10

#State space and Control space
V = FunctionSpace(mesh, 'CG', 1)

# initial guess for control
m = interpolate(Expression("x[0]", degree=1), V)

#state equation
u_ = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u_), grad(v))*dx
L = m*v*dx
bc = DirichletBC(V, 0., 'on_boundary')
u = Function(V, name='State')
solve(a == L, u, bc)

ud=ue
#L2-tracking functional and Tikhonov regularization
J_form = (0.5*inner(u-ud, u-ud))*dx + alpha/2*m**2*dx
J = assemble(J_form)

#dolfin-adjoint stuff
control = Control(m)
rf = ReducedFunctional(J, control)
dJdm = rf.derivative()

#L2-Riesz representative
RdJ = Function(V)
RdJ_ = TrialFunction(V)
a_R = RdJ_*v*dx
L_R = dJdm*v*dx

Jk = [assemble(J_form)]
mDiff = []
RdJk = []
iter = 0

iterTol = 1e-05; maxIter = 25; dispOutput = True

while Jk[-1] > iterTol and iter < maxIter:
    iter += 1

    # compute Riesz representative
    solve(a_R == L_R, RdJ)

    # update control
    m.assign(m - 100*RdJ, annotate=False)

    # solve state eqn and evaluate objective
    solve(a == L, u, bc)
    Jk.append(assemble(J_form))

    if dispOutput:
        print ('k = ' + str(iter) + ' | J = ' + str(Jk[-1]) + ' | norm(m) = ' 
            + str(norm(m, 'H1')) + ' | norm(R(dJ)) = ' + str(norm(dJdm, 'H1')))
