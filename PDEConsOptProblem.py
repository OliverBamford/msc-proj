from fenics import *
import matplotlib.pyplot as plt
import numpy as np

class PDEConsOptProblem:
    def __init__(self, N, p, ue = Expression('sin(pi*x[0])*sin(pi*x[1])', degree=3), alpha = 1e-07):
        """
        Sets up the 'hello world' PDE-constrained optimisation problem
        
        Inputs:
        N: number of finite elements in mesh
        p: order of function space
        ue: Desired distribution (UFL expression)
        alpha: regularisation parameter
        """
        mesh = UnitSquareMesh(N,N)
        V = FunctionSpace(mesh, "CG", p)
        self.ud = interpolate(ue , V)
        
        self.lmbd = interpolate(Constant(1.0), V)
        self.m = interpolate(Constant(1.0), V)
        self.mt = Function(V)
        self.RdJ = Function(V)
        self.u = Function(V)
        self.u_k = Function(V)
        self.alpha = alpha
        
        self.bc = DirichletBC(V, 0., "on_boundary")
        
        v = TestFunction(V)
        #form of state equation
        u_ = TrialFunction(V)
        self.F = inner(grad(u_), grad(v))*dx - self.m*v*dx
        #form of adjoint
        lmbd_ = TrialFunction(V)
        self.F_adj = inner(grad(lmbd_), grad(v))*dx + (self.u - self.ud)*v*dx
        # form of dJ = (RdJ, v)
        RdJ_ = TrialFunction(V)
        self.F_R = RdJ_*v*dx - (self.alpha*self.m - self.lmbd)*v*dx
        #form of objective functional
        #self.J_form = 0.5*((self.u - ud)**2 + self.alpha*self.m**2)*dx
        
    def solve_state(self):
        a,L = lhs(self.F), rhs(self.F)
        solve(a == L, self.u, self.bc)

    def solve_adjoint(self):
        a,L = lhs(self.F_adj), rhs(self.F_adj)
        solve(a == L, self.lmbd, self.bc)

    def compute_RieszRep(self):
        self.solve_state()
        self.solve_adjoint()
        a,L = lhs(self.F_R), rhs(self.F_R)
        solve(a == L, self.RdJ)

    def step_SD(self, step):
        self.m.assign(self.m - step*self.RdJ)
    
    def J(self, m):
        return 0.5*((self.u - self.ud)**2 + self.alpha*m**2)*dx
        
    def eval_J(self, m):
        self.solve_state()
        return assemble(self.J(m))
    
    def solveSD(self, step = 500., iterTol = 1.0e-5, maxIter = 25,  
                        dispOutput = False, writeData = False, filePath = 'solution-data/PDEOptSD'):
        """
        Solves the PDE-constrained opt. problem using steepest descent (SD)
        
        Inputs:
        step: initial SD step-size (will be reduced to satisfy Armijo condition)
        iterTol: Iterations stop when J < iterTol. Default: 1e-5
        maxIter: Maximum number of iterations
        dispOutput (bool): display iteration differences and objective values at each iteration
        writeData (bool): write solution and convergence data to files
        filePath: Path AND name of files WITHOUT file extension
        
        Outputs:
        [u: optimal state function
        m: optimal control function
        lmbd: Lagrange multiplier]
        [mDiff: differences between iterative solutions (in H1 norm) at each iteration
        Jk: objective value at each iteration
        RdJk: H1 norm of Riesz rep. of dJ at each iteration (SD direction)
        NOT IMPLEMENTED: refErr: H1 norms ||m_k-m_ref||. 
                                Will be an empty array if calculateRef method has not been run]
        
        Saved data:
        u saved to <filePath>_u.pvd
        m saved to <filePath>_m.pvd
        lmbd saved to <filePath>_lmbd.pvd
        Convergence data saved to <filePath>.csv:
            column 0: iterate differences
        """
        # perform one step outside of loop to ensure intial values satisfy constraints
        Jk = [self.eval_J(self.m)]
        mDiff = []
        RdJk = []
        iter = 0
        while Jk[-1] > iterTol and iter < maxIter:
            iter += 1
            self.compute_RieszRep()
            # trial step
            self.mt.assign(self.m - step*self.RdJ)
            # Frechet derivative of J at point m (previous iterate) in direction GJ, used for b-Armijo
            armijo = assemble(-(self.alpha*self.m - self.lmbd)*self.RdJ*dx)
            Jt = self.eval_J(self.m)
            # require sufficent decrease (Armijo condition)
            while Jt > (Jk[-1] + 0.1*step*armijo) and step > 1e-20 and iter > 1:
                step = 0.75*step
                # trial step with smaller step-size
                self.mt.assign(self.m - step*self.RdJ)
                Jt = self.eval_J(self.mt)
                print 'Step-size set to: ' + str(step)
                print 'J_trial = ' + str(Jt)
            if step > 1e-20:
                # step successful, update control
                mDiff.append(errornorm(self.mt, self.m, 'H1'))
                RdJk.append(norm(self.RdJ, 'H1'))
                self.step_SD(step)
                Jk.append(Jt)
            else:
                print 'Step-size reduced below threshold, convergence failed (?)'
                
            if dispOutput:
                print ('k = ' + str(iter) + ' | J = ' + str(Jk[-1]) + ' | norm(m) = ' 
                    + str(norm(self.m, 'H1')) + ' | norm(R(dJ)) = ' + str(norm(self.RdJ, 'H1')))
        # remove initial value
        Jk.pop(0)
        
        if writeData:
            # save solution
            solution = File(filePath + '_u.pvd')
            solution << self.u
            solution = File(filePath + '_m.pvd')
            solution << self.m
            solution = File(filePath + '_lmbd.pvd')
            solution <<self. lmbd
            # save convergence data
            convergenceData = [mDiff, Jk, RdJk, refErr]
            np.savetxt(filePath + '.csv', convergenceData)
            
        return [self.u, self.m, self.lmbd], [mDiff, Jk, RdJk] #, refErr]

class nonlinPDECOP:
    def __init__(self, N, p, ue = Expression('sin(pi*x[0])*sin(pi*x[1])', degree=3), alpha = 1e-07):
        """
        Sets up the 'hello world' PDE-constrained optimisation problem
        
        Inputs:
        N: number of finite elements in mesh
        p: order of function space
        ue: Desired distribution (UFL expression)
        alpha: regularisation parameter
        """
        mesh = UnitSquareMesh(N,N)
        V = FunctionSpace(mesh, "CG", p)
        self.ud = interpolate(ue , V)
        
        self.lmbd = interpolate(Constant(1.0), V)
        self.m = interpolate(Constant(1.0), V)
        self.mt = Function(V)
        self.RdJ = Function(V)
        self.u = Function(V)
        self.u_k = Function(V)
        self.du = Function(V)
        self.alpha = alpha
        
        # set up BCs on left and right
        # lambda functions ensure the boundary methods take two variables
        self.B1 = DirichletBC(V, Constant(0.0), lambda x, on_boundary : self.left_boundary(x, on_boundary)) # u(0) = 0
        self.B2 = DirichletBC(V, Constant(1.0), lambda x, on_boundary : self.right_boundary(x, on_boundary)) # u(1) = 1
        self.B2du = DirichletBC(V, Constant(0.0), lambda x, on_boundary : self.right_boundary(x, on_boundary))
        self.bcdu = [self.B1, self.B2du] # bcs for du variational problem
        self.bc = DirichletBC(V,0.,"on_boundary") # bcs for adjoint problem
        v = TestFunction(V)
          
        #form of state equation
        # construct initial guess (solution to state eqn with q(u) = 1)
        u_k_ = TrialFunction(V)
        a0du = inner(grad(u_k_), grad(v))*dx
        f = Constant(0.0)
        L0du = f*v*dx
        solve(a0du == L0du, self.u_k, [self.B1, self.B2])
        
        # construct state eqn in du          
        du_ = TrialFunction(V) # newton step
        self.adu = (inner(self.q(self.u_k)*grad(du_),grad(v)) + inner(self.dqdu(self.u_k)*du_*grad(self.u_k),grad(v)))*dx     
        self.Ldu = -inner(self.q(self.u_k)*grad(self.u_k),grad(v))*dx + self.m*v*dx
        
        #form of adjoint
        lmbd_ = TrialFunction(V)
        self.F_adj = inner(grad(lmbd_), grad(v))*dx + (self.u - self.ud)*v*dx
        # form of dJ = (RdJ, v)
        RdJ_ = TrialFunction(V)
        self.F_R = RdJ_*v*dx - (self.alpha*self.m - self.lmbd)*v*dx
        #form of objective functional
        #self.J_form = 0.5*((self.u - ud)**2 + self.alpha*self.m**2)*dx
        
    def left_boundary(self, x, on_boundary):
            return on_boundary and abs(x[0]) < 1E-14
    def right_boundary(self, x, on_boundary):
            return on_boundary and abs(x[0]-1) < 1E-14
    def q(self, u):
            return (1+u)**2
    def dqdu(self, u):
            return 2*(1+u)
            
    def solve_state(self, iterTol = 1.0e-6, maxIter = 25, dispOutput = False):
        """
        Solves the state equation using Newton iterations. Initial guess for
        first solve is calculated in __init__, value of u from last m-step
        is used thereafter.
        
        Inputs:
        iterTol: Iterations stop when |u_(k) - u_(k-1)| < iterTol. Default: 1e-5
        maxIter: Maximum number of iterations
        dispOutput(True/False): display iteration differences and exact errors at each iteration
        writeData(True/False): write solution and convergence data to files
        filePath: Path AND name of files WITHOUT file extension
        
        Outputs:
        u: solution to PDE
        iterDiffArray: Differences between iterative solutions (in H1 norm) at each iteration
        exactErrArray: Exact errors (in H1 norm) at each iteration
        
        Saved data:
        FEniCS solution saved to <filePath>.pvd
        Convergence data saved to <filePath>.csv:
            column 0: iterate differences
            column 1: exact errors
        """
        
        itErr = 1.0
        iterDiffArray = []
        iter = 0
        while itErr > iterTol and iter < maxIter:
            iter += 1
            
            solve(self.adu == self.Ldu, self.du, self.bcdu)
            self.u_k.assign(self.u + self.du)
            
            # calculate iterate difference and exact error in L2 norm
            itErr = errornorm(self.u_k, self.u, 'H1')
            #exErr = errornorm(self.uExpr, u, 'H1')
            iterDiffArray.append(itErr) # fill arrays with error data
            
            if dispOutput:
                print('k = ' + str(iter) + ' | u-diff =  ' + str(itErr))
            self.u.assign(self.u_k)

    def solve_adjoint(self):
        a,L = lhs(self.F_adj), rhs(self.F_adj)
        solve(a == L, self.lmbd, self.bc)

    def compute_RieszRep(self):
        self.solve_state(dispOutput=True)
        self.solve_adjoint()
        a,L = lhs(self.F_R), rhs(self.F_R)
        solve(a == L, self.RdJ)

    def step_SD(self, step):
        self.m.assign(self.m - step*self.RdJ)
    
    def J(self, m):
        return 0.5*((self.u - self.ud)**2 + self.alpha*m**2)*dx
        
    def eval_J(self, m):
        self.solve_state(dispOutput=True)
        return assemble(self.J(m))
    
    def solveSD(self, step = 500., iterTol = 1.0e-5, maxIter = 25,  
                        dispOutput = False, writeData = False, filePath = 'solution-data/PDEOptSD'):
        """
        Solves the PDE-constrained opt. problem using steepest descent (SD)
        
        Inputs:
        step: initial SD step-size (will be reduced to satisfy Armijo condition)
        iterTol: Iterations stop when J < iterTol. Default: 1e-5
        maxIter: Maximum number of iterations
        dispOutput (bool): display iteration differences and objective values at each iteration
        writeData (bool): write solution and convergence data to files
        filePath: Path AND name of files WITHOUT file extension
        
        Outputs:
        [u: optimal state function
        m: optimal control function
        lmbd: Lagrange multiplier]
        [mDiff: differences between iterative solutions (in H1 norm) at each iteration
        Jk: objective value at each iteration
        RdJk: H1 norm of Riesz rep. of dJ at each iteration (SD direction)
        NOT IMPLEMENTED: refErr: H1 norms ||m_k-m_ref||. 
                                Will be an empty array if calculateRef method has not been run]
        
        Saved data:
        u saved to <filePath>_u.pvd
        m saved to <filePath>_m.pvd
        lmbd saved to <filePath>_lmbd.pvd
        Convergence data saved to <filePath>.csv:
            column 0: iterate differences
        """
        # perform one step outside of loop to ensure intial values satisfy constraints
        Jk = [self.eval_J(self.m)]
        mDiff = []
        RdJk = []
        iter = 0
        while Jk[-1] > iterTol and iter < maxIter:
            iter += 1
            self.compute_RieszRep()
            # trial step
            self.mt.assign(self.m - step*self.RdJ)
            # Frechet derivative of J at point m (previous iterate) in direction GJ, used for b-Armijo
            armijo = assemble(-(self.alpha*self.m - self.lmbd)*self.RdJ*dx)
            Jt = self.eval_J(self.m)
            # require sufficent decrease (Armijo condition)
            while Jt > (Jk[-1] + 0.1*step*armijo) and step > 1e-20 and iter > 1:
                step = 0.75*step
                # trial step with smaller step-size
                self.mt.assign(self.m - step*self.RdJ)
                Jt = self.eval_J(self.mt)
                print 'Step-size set to: ' + str(step)
                print 'J_trial = ' + str(Jt)
            if step > 1e-20:
                # step successful, update control
                mDiff.append(errornorm(self.mt, self.m, 'H1'))
                RdJk.append(norm(self.RdJ, 'H1'))
                self.step_SD(step)
                Jk.append(Jt)
            else:
                print 'Step-size reduced below threshold, convergence failed (?)'
                
            if dispOutput:
                print ('k = ' + str(iter) + ' | J = ' + str(Jk[-1]) + ' | norm(m) = ' 
                    + str(norm(self.m, 'H1')) + ' | norm(R(dJ)) = ' + str(norm(self.RdJ, 'H1')))
        # remove initial value
        Jk.pop(0)
        
        if writeData:
            # save solution
            solution = File(filePath + '_u.pvd')
            solution << self.u
            solution = File(filePath + '_m.pvd')
            solution << self.m
            solution = File(filePath + '_lmbd.pvd')
            solution <<self. lmbd
            # save convergence data
            convergenceData = [mDiff, Jk, RdJk, refErr]
            np.savetxt(filePath + '.csv', convergenceData)
            
        return [self.u, self.m, self.lmbd], [mDiff, Jk, RdJk]
"""from fenics_adjoint import *
import moola

class nonlinearPDECOP:
    def __init__(self, N, p, ue = Expression('sin(pi*x[0])*sin(pi*x[1])', degree=3), alpha = 1e-07):
        mesh = UnitSquareMesh(N,N)
        V = FunctionSpace(mesh, 'CG', p)
        W = FunctionSpace(mesh, 'CG', p)
        
        # initial guess for control
        self.m = interpolate(Expression('pi*x[0]*pi*x[1]', degree=1), W)
        self.u = Function(V, name='State')
        v = TestFunction(V)
        
        # solve state equation 
        self.F = (inner(grad(self.u), grad(v)) - self.m*v)*dx
        self.bc = DirichletBC(V, 0., 'on_boundary')
        # dolfin_adjoint automatically saves iterates for use in control problem
        solve(self.F == 0, self.u, self.bc)
        
        self.ud = interpolate(ue, V)
        self.alpha = alpha
        
        J = assemble((0.5*inner(self.u-self.ud, self.u-self.ud))*dx + alpha/2*self.m**2*dx)
        control = Control(m)
        rf = ReducedFunctional(J, control)
        
 """