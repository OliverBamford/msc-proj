from fenics import *
import matplotlib.pyplot as plt
import numpy as np

class PDEConsOpt:
    def __init__(self, N, p, ud = Expression('sin(pi*x[0])*sin(pi*x[1])', degree=3), alpha = 1e-07):
        """
        Sets up the PDE-constrained optimisation problem with Lagrangian L
        
        Inputs:
        N: number of finite elements in mesh
        p: order of function space
        """
        self.N = N
        self.p = p
        self.ud = ud
        self.alpha = alpha
        
    def solveAuto(self):
        """
        Solves problem using FEniCS automatic solver (BROKEN)
        """
        u = self.u_k
        ud = self.ud
        m = self.m_k
        lmbd = self.lmbd_k
        
        self.L = (0.5*inner(u-ud, u-ud)*dx
                + 0.5*self.alpha*inner(m, m)*dx
                + inner(grad(u), grad(lmbd))*dx
                - m*lmbd*dx)
        
        
        self.F = derivative(self.L, self.z, TestFunction(self.Z))
        solve(self.F == 0, self.z, self.bcs)
        
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
        
        mDiffArray = [] 
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
            L = -(self.alpha*m_k - lmbd_k)*v*dx
            GJ = Function(M)
            solve(a == L, GJ)
            
            du = Function(U)
            du.assign(u_k-ud)
            J = 0.5*norm(du, 'L2')**2 + 0.5*alpha*norm(m_k, 'L2')**2 # current J value
            m.assign(m_k - srch*GJ) # trial m iterate
            Jk = 0.5*norm(du, 'L2')**2 + 0.5*alpha*norm(m, 'L2')**2 # trial J value
            
            # Frechet derivative of J at point m_k in direction GJ, used for b-Armijo
            # integrand = -alpha*m_k*GJ*dx
            # armijo = -alpha*inner(m_k,GJ)
            
            # begin line-search
            while Jk > J and srch > 1e-20: # ensure decrease (temporary until I can implement b-Armijo)
                srch = 0.5*srch
                m.assign(m_k - srch*GJ)
                Jk = 0.5*norm(du, 'L2')**2 + 0.5*alpha*norm(m, 'L2')**2
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
                
                
                if dispOutput:
                    print 'm-diff = ' + str(mDiff)  + ' | m-norm = ' + str(mNorm)
                    nGJ = norm(GJ, 'L2')
                    print 'J = ' + str(Jk) + '|  ||grad(J)|| = ' + str(nGJ)
                    
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
        
        du = Function(U)
        du.assign(u_k-ud)
        J = 0.5*norm(du, 'L2')**2 + 0.5*alpha*norm(m_k, 'L2')**2
        print 'Iterations terminated with J = ' + str(J) + ' and dJ = ' + str(nGJ)
        
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
            
        return u_k, m_k, lmbd_k, mDiffArray