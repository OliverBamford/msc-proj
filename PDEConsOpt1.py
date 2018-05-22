from fenics import *
import matplotlib.pyplot as plt
import numpy as np

class PDEConsOpt:
    def __init__(self, N, p):
        """
        Sets up the PDE-constrained optimisation problem with Lagrangian L
        
        Inputs:
        N: number of finite elements in mesh
        p: order of function space
        """
        alpha = 1e-07
        
        mesh = UnitSquareMesh(N,N)
        Z = VectorFunctionSpace(mesh, 'CG', p, dim=3)
        z = Function(Z)
        (u, lmbd, m) = split(z)
        
        self.bcs = [DirichletBC(Z.sub(0), 0, "on_boundary"),
               DirichletBC(Z.sub(1), 0, "on_boundary")]
        
        dist = Expression('sin(15*x[0]) + cos(20*x[1])', degree=3)
        ud = interpolate(dist, Z.sub(0).collapse())
        
        self.ud = ud
        self.u = u
        self.lmbd = lmbd
        self.m = m
        self.Z = Z
        self.z = z        
        
    def solveAuto(self):
        """
        Solves problem using FEniCS automatic solver
        """
        u = self.u
        ud = self.ud
        m = self.m
        lmbd = self.lmbd
        
        self.L = (0.5*inner(u-ud, u-ud)*dx
                + 0.5*alpha*inner(m, m)*dx
                + inner(grad(u), grad(lmbd))*dx
                - m*lmbd*dx)
        self.F = derivative(self.L, self.z, TestFunction(self.Z))
        solve(self.F == 0, self.z, self.bcs)
    
    def solveNewton(self, iterTol = 1.0e-5, maxIter = 25, dispOutput = False, writeData = True, filePath = 'solution-data/PDEOptNewton'):
        u_k = self.u
        lmbd = self.lmbd
        m = self.m
        ud = self.ud
        
        v = TestFunction(self.Z)
        
        # construct lagrangian
        Lag = (0.5*inner(u_k-ud, u_k-ud)*dx
            + 0.5*alpha*inner(m, m)*dx
            + inner(grad(u_k), grad(lmbd))*dx
            - m*lmbd*dx)     
        F = derivative(Lag, self.z, v)
        FH = derivative(F, self.z, v)
        # construct a == L for Newton iterations
        du = TrialFunction(self.Z)
        L = inner(FH, du)*v*dx
        a = inner(F,v)*dx
        
        # construct initial guess (u = ud)
        u_k = ud
        
        du = Function(V)
        u = Function(V)
        itErr = 1.0
        iterDiffArray = []
        exactErrArray = []
        iter = 0
        while itErr > iterTol and iter < maxIter:
            iter += 1
            
            solve(a == L, du, self.bcs)
            u.vector()[:] = u_k.vector() + du.vector()
            
            # calculate iterate difference and exact error in H1 norm
            itErr = errornorm(u_k, u, 'H1')
            # exErr = errornorm(self.uExpr, u, 'H1')
            iterDiffArray.append(itErr) # fill arrays with error data
            exactErrArray.append(exErr)    
            
            if dispOutput:
                print('k = ' + str(iter) + ' | u-diff =  ' + str(itErr) + ', exact error = ' + str(exErr))
            u_k.assign(u)
        
        if writeData:
            # save solution
            solution = File(filePath + '.pvd')
            solution << u_k
            # save convergence data
            convergenceData = [iterDiffArray, exactErrArray]
            np.savetxt(filePath + '.csv', convergenceData)
            
        # save data to object
        self.newtonSol = u_k
        self.newtonIterDiff = iterDiffArray
        self.newtonExactErr = exactErrArray  
        return [u_k, iterDiffArray, exactErrArray]               