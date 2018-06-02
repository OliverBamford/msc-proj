from fenics import *
import matplotlib.pyplot as plt
import numpy as np

class PDEConsOpt:
    def __init__(self, N, p, ue = Expression('sin(pi*x[0])*sin(pi*x[1])', degree=3), alpha = 1e-07):
        """
        Sets up the 'hello world' PDE-constrained optimisation problem
        
        Inputs:
        N: number of finite elements in mesh
        p: order of function space
        ud: Desired distribution (UFL expression)
        alpha: regularisation parameter
        """
        
        # set up problem (parts common to all iterative methods)
        mesh = UnitSquareMesh(N, N)
        V = FunctionSpace(mesh, 'CG', p)
        bcs = DirichletBC(V, 0, "on_boundary")
        # interpolate ud over function space
        ud = interpolate(ue , V)
        # initialise problem variables
        u = TrialFunction(V)
        lmbd = TrialFunction(V)
        GJ = TrialFunction(V)
        v = TestFunction(V)
        
        self.N = N
        self.p = p
        self.ue=ue
        self.ud = ud
        self.alpha = alpha
        self.mesh = mesh
        self.V = V
        self.bcs = bcs
        self.u = u
        self.lmbd = lmbd
        self.GJ = GJ
        self.v = v
    def calculateRef(self):
        # construct mixed function space
        mesh = self.mesh
        alpha = self.alpha
        ue = self.ue
        
        P2 = VectorElement("Lagrange", mesh.ufl_cell(), dim=2,degree=2)
        P1 = FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)
        TH = P2*P1
        H = FunctionSpace(mesh, TH)
        U = TrialFunction(H)
        ul,m = split(U)
        u, lmbd = split(ul)
        phi,psi,chi = TestFunction(H)
        ud = interpolate(ue , H.sub(0).sub(0).collapse()) # interpolate over u-space
        
#        A = assemble(inner(grad(u), grad(phi))*dx) # assemble mass matrix for u,lmbd
#        M = assemble(inner(grad(m), grad(chi))*dx) # assemble mass matrix for m
#        mu = assemble(inner(ud, phi)*dx) # assemble RHS (for adj. equation)
        
        #F = (inner(u,phi)-m*phi)*dx + (inner(lmbd, psi) + (u-ud)*psi)*dx + (alpha*m - lmbd)*chi*dx
        a = (inner(u,phi)-m*phi)*dx + (inner(lmbd, psi) + u*psi)*dx + (alpha*m - lmbd)*chi*dx
        L = ud*psi*dx
        
        U = Function(H)
        ul,m = split(U)
        u, lmbd = split(ul)
        solve(a == L, U, solver_parameters={"linear_solver": "lu"})
        
        return u, m, lmbd
    def solveSD(self, srch = 100, iterTol = 1.0e-5, maxIter = 25,  
                        dispOutput = False, writeData = False, filePath = 'solution-data/PDEOptSD'):
        """
        Solves the PDE-constrained opt. problem using steepest descent (SD)
        
        Inputs:
        srch: initial SD step-size (will be reduced to satisfy Armijo condition)
        iterTol: Iterations stop when J < iterTol. Default: 1e-6
        maxIter: Maximum number of iterations
        dispOutput(bool): display iteration differences and objective values at each iteration
        writeData(True/False): write solution and convergence data to files
        filePath: Path AND name of files WITHOUT file extension
        
        Outputs:
        u: optimal state function
        m: optimal control function
        lmbd: Lagrange multiplier
        mDiffArray: differences between iterative solutions (in H1 norm) at each iteration
        J: objective value at each iteration
        nGJ: H1 norm of Riesz rep. of dJ at each iteration (SD direction)
        
        Saved data:
        u saved to <filePath>_u.pvd
        m saved to <filePath>_m.pvd
        lmbd saved to <filePath>_lmbd.pvd
        Convergence data saved to <filePath>.csv:
            column 0: iterate differences
        """
        N = self.N
        p = self.p
        ud = self.ud
        alpha = self.alpha
        mesh = self.mesh
        V = self.V
        bcs = self.bcs
        u = self.u
        lmbd = self.lmbd
        GJ = self.GJ
        v = self.v
        
        # initialise arrays to store convergence data, these initial values will be removed
        mDiffArray = [1e99]
        J = [1e99]
        nGJ = [1e99]
        ndu= [1e99]
        iter = 0
        
        # initial guesses
        lmbd_k = interpolate(Constant(1.0), V)
        m_k = interpolate(Constant(1.0), V)
        # initialise functions
        u_k = Function(V)
        GJ_k = Function(V)
        m = Function(V)
        du = Function(V)
        # begin steepest descent
        while J[-1] > iterTol and iter < maxIter:
            iter += 1
            
            # find u which satisfies state equation
            State = inner(grad(v),grad(u))*dx
            L = m_k*v*dx
            solve(State == L, u_k, bcs)
            
            # find lambda that satisfies adjoint equation
            Adj = inner(grad(v),grad(lmbd))*dx
            L = -(u_k-ud)*v*dx
            solve(Adj == L, lmbd_k, bcs)
    
            # find the Riesz rep. of dJ 
            a = GJ*v*dx
            L = (alpha*m_k - lmbd_k)*v*dx
            solve(a == L, GJ_k)
            
            du.assign(u_k-ud)
            ndu.append(0.5*norm(du, 'L2')**2)
            m.assign(m_k - srch*GJ_k) # trial m iterate
            Jk = ndu[-1] + 0.5*alpha*norm(m, 'L2')**2 # objective value with trial iterate
            
            # Frechet derivative of J at point m_k in direction GJ, used for b-Armijo
            armijo = assemble(-(alpha*m_k - lmbd_k)*GJ_k*dx)
            
            # begin line-search
            while Jk > J[-1] + 0.01*srch*armijo and srch > 1e-20: # impose Armijo condition ==
                srch = 0.5*srch
                m.assign(m_k - srch*GJ_k)
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
                nGJ.append(norm(GJ_k, 'L2'))
                
                if dispOutput:
                    print 'm-diff = ' + str(mDiff)  + ' | m-norm = ' + str(mNorm)
                    print 'J = ' + str(Jk) + '|  ||grad(J)|| = ' + str(nGJ[-1])
                    
        # update u and lambda using final m value
        State = inner(grad(v),grad(u))*dx
        L = m_k*v*dx
        solve(State == L, u_k, bcs)
        
        Adj = inner(grad(v),grad(lmbd))*dx
        L = -(u_k-ud)*v*dx
        solve(Adj == L, lmbd_k, bcs)
        
        # find the Riesz rep. of dJ 
        a = GJ*v*dx
        L = (alpha*m_k - lmbd_k)*v*dx 
        solve(a == L, GJ_k)
        
        du = Function(V)
        du.assign(u_k-ud)
        J.append(0.5*norm(du, 'L2')**2 + 0.5*alpha*norm(m_k, 'L2')**2)
        print 'Iterations terminated with J = ' + str(J[-1]) + ' and dJ = ' + str(nGJ[-1])
        
        mDiffArray.pop(0)
        J.pop(0)
        nGJ.pop(0)
        ndu.pop(0)
        
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
        
        return u_k, m_k, lmbd_k, mDiffArray, J, nGJ