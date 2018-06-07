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
        [mDiffArray: differences between iterative solutions (in H1 norm) at each iteration
        J: objective value at each iteration
        nGJ: H1 norm of Riesz rep. of dJ at each iteration (SD direction)
        refErr: H1 norms ||m_k-m_ref||. Will be an empty array if calculateRef method has not been run]
        
        Saved data:
        u saved to <filePath>_u.pvd
        m saved to <filePath>_m.pvd
        lmbd saved to <filePath>_lmbd.pvd
        Convergence data saved to <filePath>.csv:
            column 0: iterate differences
        """
        # perform one step outside of loop to ensure intial values satisfy constraints
        Jk = [self.eval_J(self.m)]
        iter = 0
        while Jk[-1] > iterTol and iter < maxIter:
            iter += 1
            
            self.compute_RieszRep()
#            self.mt.assign(self.m - step*self.RdJ) # trial step
            self.mt.assign(self.m - step*self.RdJ)
            # Frechet derivative of J at point m (previous iterate) in direction GJ, used for b-Armijo
            armijo = assemble(-(self.alpha*self.m - self.lmbd)*self.RdJ*dx)
            Jt = self.eval_J(self.m)
            print 'J_trial = ' + str(Jt) + ' | J_target = ' + str(Jk[-1] + 0.1*step*armijo)
            while Jt > (Jk[-1] + 0.1*step*armijo) and step > 1e-20 and iter > 1:
                step = 0.75*step
                # trial step with smaller step-size
                self.mt.assign(self.m - step*self.RdJ)
                Jt = self.eval_J(self.mt)
                print 'Step-size set to: ' + str(step)
                print 'J_trial = ' + str(Jt)
            if step > 1e-20:
                # step successful, update control
                self.step_SD(step)
                Jk.append(Jt)
            else:
                print 'Step-size reduced below threshold, convergence failed (?)'
                
            if dispOutput:
                print ('k = ' + str(iter) + ' | J = ' + str(Jk[-1]) + ' | norm(m) = ' 
                    + str(norm(self.m, 'H1')) + ' | norm(R(dJ)) = ' + str(norm(self.RdJ, 'H1')))
        # remove initial value
        Jk.pop(0)