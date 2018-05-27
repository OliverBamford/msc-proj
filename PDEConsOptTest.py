from fenics import *
import matplotlib.pyplot as plt
import numpy as np

N = 100
p = 1
alpha = 1e-07

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
srch = 100 # step size

u_k.assign(ud) # initial guesses
lmbd_k = interpolate(Constant(1.0), LMBD)
m_k = interpolate(Constant(1.0), M) 

m = Function(M)
# begin steepest descent
while itErr > 1e-06 and iter < 2500:
    iter += 1
    
    # find u which satisfies state equation
    u = TrialFunction(U)
    v = TestFunction(U)
    State = inner(grad(v),grad(u))*dx
    L = -m_k*v*dx
    u = Function(U)
    solve(State == L, u, bcs[0])
    uNorm = norm(u, 'H1')
    uDiff = errornorm(u_k, u, 'H1')
    print 'u-diff = ' + str(uDiff)  + ' | u-norm = ' + str(uNorm)
    u_k.assign(u)
    
    # find lambda that satisfies adjoint equation
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
    L = -(alpha*m_k - lmbd_k)*v*dx
    GJ = Function(M)
    solve(a == L, GJ)
    
    du = Function(U)
    du.assign(u_k-ud)
    J = 0.5*norm(du, 'L2')**2 - 0.5*alpha*norm(m_k, 'L2')**2
    m.assign(m_k - srch*GJ)
    Jk = 0.5*norm(du, 'L2')**2 - 0.5*alpha*norm(m, 'L2')**2
    
    # Frechet derivative of J at point m_k in direction GJ, used for b-Armijo
    # integrand = -(alpha*m_k - lmbd_k)*GJ*dx
    
    # find step-size
    while Jk > 0.9999*J: # ensure 1% decrease (temporary until I can implement b-Armijo)
        srch = 0.9*srch
        m.assign(m_k - srch*GJ)
        Jk = 0.5*norm(du, 'L2')**2 - 0.5*alpha*norm(m, 'L2')**2
        print 'Step-size set to: ' + str(srch)
    
    mNorm = norm(m, 'H1')
    mDiff = errornorm(m_k, m, 'H1')
    print 'm-diff = ' + str(mDiff)  + ' | m-norm = ' + str(mNorm)
    # update m using steepest descent
    m_k.assign(m)

    J = 0.5*norm(du, 'L2')**2 - 0.5*alpha*norm(m_k, 'L2')**2
    nGJ = norm(GJ, 'L2')
    print 'J = ' + str(J) + '|  ||grad(J)|| = ' + str(nGJ)
