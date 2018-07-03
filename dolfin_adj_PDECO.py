from fenics import *
import matplotlib.pyplot as plt
import numpy as np
from fenics_adjoint import *

class PDEConstrOptProblem(object):
    def __init__(self):
        mesh = UnitSquareMesh(10,10)
        V = FunctionSpace(mesh, "CG", 1)

        X = SpatialCoordinate(mesh)
        u_t = sin(pi*X[0])*sin(pi*X[1])
        self.alpha = 1e-07

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
                            
        Jk = [self.eval_J()]
        iter = 0
        if solver == 'SD':
            while Jk[-1] > iterTol and iter < maxIter:
                iter += 1
                # compute Riesz rep of dJ
                self.compute_RieszRep()
                # update control
                self.update_control(step)
                # solve state eqn and evaluate objective
                Jk.append(self.eval_J())
            
                if dispOutput:
                    print ('k = ' + str(iter) + ' | J = ' + str(Jk[-1]) + ' | norm(m) = ' 
                        + str(norm(self.m, 'H1')) + ' | norm(R(dJ)) = ' + str(norm(self.dm, 'H1')))
                            
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


