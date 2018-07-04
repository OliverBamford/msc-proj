from fenics import *
import matplotlib.pyplot as plt
import numpy as np
from fenics_adjoint import *

class PDEConstrOptProblem(object):
    def __init__(self, N=10, p=1, ue = Expression('sin(pi*x[0])*sin(pi*x[1])', degree=3), alpha = 1e-07):
        """
        Sets up the 'hello world' PDE-constrained optimisation problem
        
        Inputs:
        N: number of finite elements in mesh
        p: order of function space
        ue: Desired distribution (UFL expression)
        alpha: regularisation parameter
        """
        mesh = UnitSquareMesh(N, N)
        V = FunctionSpace(mesh, "CG", p)

        X = SpatialCoordinate(mesh)
        #u_t = interpolate(ue, V) #TODO: find out why this implemetation gives
                                    # better convergence
        u_t = sin(pi*X[0]) * sin(pi*X[1]) 
        self.alpha = alpha

        self.m = interpolate(Constant(1.0), V)
        self.m.rename("Control", "")
        self.dm = Function(V)
        self.u = Function(V, name='State')

        v = TestFunction(V)

        #state equation
        u_ = TrialFunction(V)
        self.F = inner(grad(u_), grad(v))*dx - self.m*v*dx
        self.bc = DirichletBC(V, 0., "on_boundary")
        
        # solve state eqn before doing anything else for dolfin-adjoint
        self.solve_state()
        
        #form of misfit functional
        self.J_form = 0.5*(inner(self.u - u_t, self.u - u_t)*dx + self.alpha*self.m**2*dx)
        J = assemble(self.J_form)
        
        #dolfin-adjoint stuff
        control = Control(self.m)
        self.rf = ReducedFunctional(J, control)

    def solve_state(self):
        solve(lhs(self.F) == rhs(self.F), self.u, self.bc)

    def eval_J(self):
        self.solve_state()
        return self.rf(self.m)
        #return assemble(self.J_form)
        
    def compute_RieszRep(self):
        self.dm = self.rf.derivative(options={"riesz_representation": "L2"})

    def update_control(self, step):
        self.m.vector()[:] -= step * self.dm.vector()[:]

    def solve(self, solver = 'SD', step = 500., iterTol = 1.0e-5, maxIter = 25,  
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
        m: optimal control function]
        [mDiff: differences between iterative solutions (in H1 norm) at each iteration
        Jk: objective value at each iteration
        RdJk: H1 norm of Riesz rep. of dJ at each iteration (SD direction)
        NOT IMPLEMENTED: refErr: H1 norms ||m_k-m_ref||. 
                                Will be an empty array if calculateRef method has not been run]
        
        Saved data:
        u saved to <filePath>_u.pvd
        m saved to <filePath>_m.pvd
        Convergence data saved to <filePath>.csv:
            column 0: iterate differences
        """
        
        self.compute_RieszRep()
        RdJk = [norm(self.dm, 'H1')]
        mDiff = [np.nan]
        Jk = [self.eval_J()]
        iter = 0
        if solver == 'SD':
            while Jk[-1] > iterTol and iter < maxIter:
                iter += 1
                # compute Riesz rep of dJ
                self.compute_RieszRep()
                # store previous iterate
                m_prev = self.m
                # update control
                self.update_control(step)
                # solve state eqn, evaluate objective and store data
                Jk.append(self.eval_J())
                RdJk.append(norm(self.dm, 'H1'))
                mDiff.append(errornorm(self.m, m_prev, 'H1'))
                if dispOutput:
                    print ('k = ' + str(iter) + ' | J = ' + str(Jk[-1]) + ' | norm(m) = ' 
                        + str(norm(self.m, 'H1')) + ' | norm(R(dJ)) = ' + str(RdJk[-1]))
        if writeData:
                # save solution
                solution = File(filePath + '_u.pvd')
                solution << self.u
                solution = File(filePath + '_m.pvd')
                solution << self.m
                # save convergence data
                convergenceData = [mDiff, Jk, RdJk, refErr]
                np.savetxt(filePath + '.csv', convergenceData)
            
        return [self.u, self.m], [mDiff, Jk, RdJk]
#P = PDEConstrOptProblem()
#
#for ii in range(5):
#    Jval[ii] = eval_J()
#    compute_RieszRep()
#
#    for jj in range(3):
#        update_control(0.1)
#        Jtemp[jj] = eval_J()
#
#    minIdx = indexOfMinimum(Jtemp)
#    update_control(-0.3+0.1*(1+minIdx))
#
#
#mDiff = []
#RdJk = []
#iter = 0


