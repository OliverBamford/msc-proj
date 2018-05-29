from fenics import *
import matplotlib.pyplot as plt
import numpy as np

class PDEConsOpt:
    def __init__(self, N, p, ud = Expression('sin(pi*x[0])*sin(pi*x[1])', degree=3), alpha = 1e-07):
        """
        Sets up the 'hello world' PDE-constrained optimisation problem
        
        Inputs:
        N: number of finite elements in mesh
        p: order of function space
        ud: Desired distribution (UFL expression)
        alpha: regularisation parameter
        """
        self.N = N
        self.p = p
        self.ud = ud
        self.alpha = alpha
        
    def solveSD(self, srch = 100, iterTol = 1.0e-5, maxIter = 25, dispOutput = False, writeData = False, filePath = 'solution-data/PDEOptSD'):
        """
        Solves the PDE-constrained opt. problem using steepest descent (SD)
        
        Inputs:
        srch: initial SD step-size
        iterTol: Iterations stop when |m_(k) - m_(k-1)| < iterTol. Default: 1e-5
        maxIter: Maximum number of iterations
        dispOutput(bool): display iteration differences and objective values at each iteration
        writeData(True/False): write solution and convergence data to files
        filePath: Path AND name of files WITHOUT file extension
        
        Outputs:
        u: optimal state function
        m: optimal control function
        lmbd: lagrange multiplier
        mDiffArray: Differences between iterative solutions (in H1 norm) at each iteration
        
        Saved data:
        u saved to <filePath>_u.pvd
        m saved to <filePath>_m.pvd
        lmbd saved to <filePath>_lmbd.pvd
        Convergence data saved to <filePath>.csv:
            column 0: iterate differences
        """
        alpha = self.alpha
        
        mesh = UnitSquareMesh(self.N, self.N)
        Z = VectorFunctionSpace(mesh, 'CG', self.p, dim=3)
        
        U = Z.sub(0).collapse()
        LMBD = Z.sub(1).collapse()
        M = Z.sub(2).collapse()
        u_k = Function(U)
        lmbd_k = Function(LMBD)
        m_k = Function(M)
        
        bcs = [DirichletBC(U, 0, "on_boundary"),
               DirichletBC(LMBD, 0, "on_boundary")]
               
        ud = interpolate(self.ud, U)
        
        mDiffArray = [1e99] 
        J = [1e99]
        nGJ = [1e99]
        iter = 0
        
        # initial guesses
        lmbd_k = interpolate(Constant(1.0), LMBD)
        m_k = interpolate(Constant(1.0), M) 
        
        m = Function(M)
        mDiff = 1.0
        # begin steepest descent
        while mDiff > iterTol and iter < maxIter:
            iter += 1
            
            # find u which satisfies state equation
            u = TrialFunction(U)
            v = TestFunction(U)
            State = inner(grad(v),grad(u))*dx
            L = -m_k*v*dx
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
            L = -(alpha*m_k - lmbd_k)*v*dx #TODO: explain minus sign
            GJ = Function(M)
            solve(a == L, GJ)            
            
            du = Function(U)
            du.assign(u_k-ud)
            ndu = 0.5*norm(du, 'L2')**2
            #J = 0.5*norm(du, 'L2')**2 + 0.5*alpha*norm(m_k, 'L2')**2 # current J value
            m.assign(m_k - srch*GJ) # trial m iterate
            Jk = ndu + 0.5*alpha*norm(m, 'L2')**2 # trial J value
            
            # Frechet derivative of J at point m_k in direction GJ, used for b-Armijo
            # integrand = -alpha*m_k*GJ*dx
            # armijo = -alpha*inner(m_k,GJ)
            
            # begin line-search
            while Jk > J[-1] and srch > 1e-20: #TODO: impose Armijo condition here
                srch = 0.5*srch
                m.assign(m_k - srch*GJ)
                Jk = ndu + 0.5*alpha*norm(m, 'L2')**2
                print 'Step-size set to: ' + str(srch)
                
            if srch < 1e-20:
                print 'Step-size below threshold, possible minimum found.'
                iter = maxIter
            else:
                mNorm = norm(m, 'H1')
                mDiff = errornorm(m_k, m, 'H1')
                mDiffArray.append(mDiff)
            
                # update m_k
                m_k.assign(m)
                
                J.append(Jk)
                nGJ.append(norm(GJ, 'L2'))
                if dispOutput:
                    print 'm-diff = ' + str(mDiff)  + ' | m-norm = ' + str(mNorm)
                    print 'J = ' + str(Jk) + '|  ||grad(J)|| = ' + str(nGJ[-1])
                    
        # update u and lambda using final m value
        u = TrialFunction(U)
        v = TestFunction(U)
        State = inner(grad(v),grad(u))*dx
        L = -m_k*v*dx
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
        
        du.assign(u_k-ud)
        J.append(0.5*norm(du, 'L2')**2 + 0.5*alpha*norm(m_k, 'L2')**2)
        print 'Iterations terminated with J = ' + str(J[-1]) + ' and dJ = ' + str(nGJ[-1])
        
        if writeData:
            # save solution
            solution = File(filePath + '_u.pvd')
            solution << u_k
            solution = File(filePath + '_m.pvd')
            solution << m_k
            solution = File(filePath + '_lmbd.pvd')
            solution << lmbd_k
            # save convergence data
            convergenceData = mDiffArray
            np.savetxt(filePath + '.csv', convergenceData)
        
        mDiffArray.pop(0)
        J.pop(0)
        nGJ.pop(0)
        return u_k, m_k, lmbd_k, mDiffArray, J, nGJ