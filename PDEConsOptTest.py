from fenics import *
import matplotlib.pyplot as plt
import numpy as np

N = 100
p = 1

alpha = 0.001

mesh = UnitSquareMesh(N,N)
Z = VectorFunctionSpace(mesh, 'CG', p, dim=3)
z = Function(Z)
#(u_k, lmbd_k, m_k) = split(z)

U = Z.sub(0).collapse()
LMBD = Z.sub(1).collapse()
M = Z.sub(2).collapse()
u_k = Function(U)
lmbd_k = Function(LMBD)
m_k = Function(M)

bcs = [DirichletBC(U, 0, "on_boundary"),
       DirichletBC(LMBD, 0, "on_boundary")]

dist = Expression('x[0]*x[1]*(x[0] - 1)*(x[1] - 1)', degree=3)
# ud = interpolate(dist, Z.sub(0).collapse())
ud = interpolate(dist, U)

itErr = 1.0  # error measure ||u-u_k||
iterDiffArray = []
exactErrArray = []   
iter = 0
srch = 1

u_k.assign(ud) # initial guesses
lmbd_k = interpolate(Constant(0.0), LMBD)
m_k = interpolate(Constant(0.0), M) 

m = Function(M)
# begin steepest descent
while itErr > 1e-06 and iter < 5:
    iter += 1
    
    # find u which satisfies PDE constraint (state equation)
    u = TrialFunction(U)
    v = TestFunction(U)
    State = inner(grad(v),grad(u))*dx
    L = -m_k*v*dx
    u = Function(U)
    solve(State == L, u, bcs[0])
    uNorm = errornorm(u, interpolate(Constant(0.0), U), 'H1')
    uDiff = errornorm(u_k, u, 'H1')
    print 'u-diff = ' + str(uDiff)  + ' | u-norm = ' + str(uNorm)
    u_k.assign(u)
    
    # update lambda from adjoint equation
    lmbd = TrialFunction(LMBD)
    v = TestFunction(LMBD)
    Adj = inner(grad(v),grad(lmbd))*dx
    L = -(u_k-ud)*v*dx
    lmbd = Function(LMBD)
    solve(Adj == L, lmbd, bcs[1])
    lmbd_k.assign(lmbd)
    
    # find the Riesz rep. of dJ 
    GJ = TrialFunction(M)
    v = TestFunction(M)
    a = GJ*v*dx
    L = (alpha*m_k - lmbd_k)*v*dx
    GJ = Function(M)
    solve(a == L, GJ)
    # update m using steepest descent
    m.assign(m_k - srch*GJ)
    mNorm = errornorm(m, interpolate(Constant(0.0), M), 'H1')
    mDiff = errornorm(m_k, m, 'H1')
    print 'm-diff = ' + str(mDiff)  + ' | m-norm = ' + str(mNorm)
    m_k.assign(m)
    
