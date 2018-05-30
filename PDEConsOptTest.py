from fenics import *
import matplotlib.pyplot as plt
import numpy as np

NN = np.arange(10, 51, 5)
p = 1
alpha = 1e-07
finalJ = []

# this loop is so that the mesh dependence of J may be plotted
for N in [100]:
    print str(N)
    mesh = UnitSquareMesh(N, N)
    Z = VectorFunctionSpace(mesh, 'CG', p, dim=3)
    
    U = Z.sub(0).collapse()
    LMBD = Z.sub(1).collapse()
    M = Z.sub(2).collapse()
    u_k = Function(U)
    lmbd_k = Function(LMBD)
    m_k = Function(M)
    
    bcs = [DirichletBC(U, 0, "on_boundary"),
           DirichletBC(LMBD, 0, "on_boundary")]
    
    ud = interpolate(Expression('sin(pi*x[0])*sin(pi*x[1])', degree=3) , U)
    
    # initialise arrays to store convergence data, these initial values will be removed
    mDiffArray = [1e99] 
    J = [1e99]
    nGJ = [1e99]
    ndu= [1e99]
    iter = 0
    
    # initial guesses
    lmbd_k = interpolate(Constant(1.0), LMBD)
    m_k = interpolate(Constant(1.0), M) 
    
    m = Function(M)
    mDiff = 1.0
    iterTol = 1e-05
    maxIter = 10
    srch = 500
    # begin steepest descent
    while mDiff > iterTol and iter < maxIter:
        iter += 1
        
        # find u which satisfies state equation
        u = TrialFunction(U)
        v = TestFunction(U)
        State = inner(grad(v),grad(u))*dx
        L = m_k*v*dx
        u = Function(U)
        solve(State == L, u, bcs[0])
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
        L = (alpha*m_k - lmbd_k)*v*dx
        GJ = Function(M)
        solve(a == L, GJ)
        
        du = Function(U)
        du.assign(u_k-ud)
        ndu.append(0.5*norm(du, 'L2')**2)
        m.assign(m_k - srch*GJ) # trial m iterate
        Jk = ndu[-1] + 0.5*alpha*norm(m, 'L2')**2 # objective value with trial iterate
        
        # Frechet derivative of J at point m_k in direction GJ, used for b-Armijo
        armijo = assemble(-(alpha*m_k - lmbd_k)*GJ*dx)
        
        # begin line-search
        while Jk > J[-1] + 0.01*srch*armijo and srch > 1e-20: # impose Armijo condition ==
            srch = 0.5*srch
            m.assign(m_k - srch*GJ)
            Jk = ndu[-1] + 0.5*alpha*norm(m, 'L2')**2
            print 'Step-size set to: ' + str(srch)
            
        if srch < 1e-20:
            print 'Step-size below threshold, convergence failed(?).'
            iter = maxIter
        else:
            mNorm = norm(m, 'H1')
            mDiff = errornorm(m_k, m, 'H1')
            mDiffArray.append(mDiff)
        
            # update m_k
            m_k.assign(m)
            
            J.append(Jk)
            nGJ.append(norm(GJ, 'L2'))
            
            print 'm-diff = ' + str(mDiff)  + ' | m-norm = ' + str(mNorm)
            print 'J = ' + str(Jk) + '|  ||grad(J)|| = ' + str(nGJ[-1])
                
    # update u and lambda using final m value
    u = TrialFunction(U)
    v = TestFunction(U)
    State = inner(grad(v),grad(u))*dx
    L = m_k*v*dx
    u = Function(U)
    solve(State == L, u, bcs[0])
    u_k.assign(u)
    
    lmbd = TrialFunction(LMBD)
    v = TestFunction(LMBD)
    Adj = inner(grad(v),grad(lmbd))*dx
    L = -(u_k-ud)*v*dx
    lmbd = Function(LMBD)
    solve(Adj == L, lmbd, bcs[1])
    lmbd_k.assign(lmbd) 
    
    du = Function(U)
    
    du.assign(u_k-ud)
    J.append(0.5*norm(du, 'L2')**2 + 0.5*alpha*norm(m_k, 'L2')**2)
    print 'Iterations terminated with J = ' + str(J[-1]) + ' and dJ = ' + str(nGJ[-1])
    
    mDiffArray.pop(0)
    J.pop(0)
    nGJ.pop(0)
    ndu.pop(0)
    
    finalJ.append(J[-1])
    
#%% PLOTTING

plt.figure(1)
plt.plot(mDiffArray, label='$||m_{k+1} - m_k||$')
plt.semilogy(ndu, label='$||u_{k+1} - u_k||$')
plt.semilogy(J, label='$J$')
plt.semilogy(nGJ, label='$R(dJ)$')
plt.title('Convergence of Steepest-Descent Iterations on "hello world!" PDECO Problem (N = 100, p = 1)')
plt.legend()

plt.figure(2)
plt.subplot(2,2,1)
plt.title('$u_d$')
plot(ud)
plt.subplot(2,2,2)
plt.title('$u_k$')
plot(u_k)
plt.subplot(2,2,3)
plt.title('$m_k$')
plot(m_k)
plt.subplot(2,2,4)
plt.title('$\lambda_k$')
plot(lmbd_k)
plt.suptitle('Solution to "hello world!" PDECO Problem After 25 SD Iterations (N = 100, p = 1)')