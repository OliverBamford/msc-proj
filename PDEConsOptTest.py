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
    V = FunctionSpace(mesh, 'CG', p)
    
    u_k = Function(V)
    lmbd_k = Function(V)
    m_k = Function(V)
    
    bcs = DirichletBC(V, 0, "on_boundary")
    ud = interpolate(Expression('sin(pi*x[0])*sin(pi*x[1])', degree=3) , V)
    
    # initial guesses
    lmbd_k = interpolate(Constant(1.0), V)
    m_k = interpolate(Constant(1.0), V)
    m = Function(V)
    
    # initialise arrays to store convergence data, these initial values will be removed
    mDiffArray = [1e99]
    J[-1] = [1e99]
    nGJ = [1e99]
    ndu= [1e99]
    iter = 0
    
    mDiff = 1.0
    iterTol = 1e-05
    maxIter = 10
    srch = 500
    # begin steepest descent
    while J[-1] > iterTol and iter < maxIter:
        iter += 1
        
        # find u which satisfies state equation
        u = TrialFunction(V)
        v = TestFunction(V)
        State = inner(grad(v),grad(u))*dx
        L = m_k*v*dx
        solve(State == L, u_k, bcs)
        
        if iter == 1:
                du = Function(V)
                du.assign(u_k-ud)
                ndu.append(0.5*norm(du, 'L2')**2)
                J = [ndu[-1] + 0.5*alpha*norm(m_k, 'L2')**2]
        
        # find lambda that satisfies adjoint equation
        lmbd = TrialFunction(V)
        v = TestFunction(V)
        Adj = inner(grad(v),grad(lmbd))*dx
        L = -(u_k-ud)*v*dx
        solve(Adj == L, lmbd_k, bcs)

        # find the Riesz rep. of dJ 
        GJ = TrialFunction(V)
        v = TestFunction(V)
        a = GJ*v*dx
        L = (alpha*m_k - lmbd_k)*v*dx
        GJ = Function(V)
        solve(a == L, GJ)
        
        du = Function(V)
        du.assign(u_k-ud)
        ndu.append(0.5*norm(du, 'L2')**2)
        m.assign(m_k - srch*GJ) # trial m iterate
        Jk = ndu[-1] + 0.5*alpha*norm(m, 'L2')**2# objective value with trial iterate
        
        # Frechet derivative of J at point m_k in direction GJ, used for b-Armijo
        armijo = assemble(-(alpha*m_k - lmbd_k)*GJ*dx)
        
        # begin line-search
#        while Jk > J[-1] + 0.1*srch*armijo and srch > 1e-20: # impose Armijo condition ==
#            srch = 0.5*srch
#            m.assign(m_k - srch*GJ)
#            Jk = ndu[-1] + 0.5*alpha*norm(m, 'L2')**2
#            print 'Step-size set to: ' + str(srch)
            
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
    u = TrialFunction(V)
    v = TestFunction(V)
    State = inner(grad(v),grad(u))*dx
    L = m_k*v*dx
    solve(State == L, u_k, bcs)
    
    lmbd = TrialFunction(V)
    v = TestFunction(V)
    Adj = inner(grad(v),grad(lmbd))*dx
    L = -(u_k-ud)*v*dx
    solve(Adj == L, lmbd_k, bcs)
    
    # find the Riesz rep. of dJ 
    GJ = TrialFunction(V)
    v = TestFunction(V)
    a = GJ*v*dx
    L = (alpha*m_k - lmbd_k)*v*dx 
    GJ = Function(V)
    solve(a == L, GJ)
    
    du = Function(V)
    
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
plt.semilogy(ndu, label='$||u_{k+1} - u_d||$')
plt.semilogy(J, label='$J$')
plt.semilogy(nGJ, label='$R(dJ)$')
plt.semilogy(refErr, label='$||m_k - m_{ref}||$')
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