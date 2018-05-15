from fenics import *
import matplotlib.pyplot as plt
import numpy as np

class nonlinearPDE1:
    def __init__(self, N, p):
        """
        Sets up the 1D PDE: -grad((1+u)^2.grad(u)) = 0
        
        Inputs:
        N: number of finite elements in mesh
        p: order of function space
        """
        # set up function space
        mesh = UnitIntervalMesh(N)
        self.V = FunctionSpace(mesh, 'CG', p)
        
        # set up BCs on left and right
        # lambda functions ensure the boundary methods take two variables
        self.B1 = DirichletBC(self.V, Constant(0.0), lambda x, on_boundary : self.left_boundary(x, on_boundary)) # u(0) = 0
        self.B2 = DirichletBC(self.V, Constant(1.0), lambda x, on_boundary : self.right_boundary(x, on_boundary)) # u(1) = 1

        # construct exact solution in C format
        self.uExpr = Expression('pow((pow(2,m+1) - 1)*x[0] + 1,(1/(m+1))) - 1', m = 2, degree=4)
        
    def left_boundary(self, x, on_boundary):
            return on_boundary and abs(x[0]) < 1E-14
    def right_boundary(self, x, on_boundary):
            return on_boundary and abs(x[0]-1) < 1E-14
    def q(self, u):
            return (1+u)**2
    def dqdu(self,u):
            return 2*(1+u)
            
    def solvePicard(self, iterTol = 1.0e-5, maxIter = 25, dispOutput = False, writeData = True, filePath = 'solution-data/PDE1Picard'):
        """
        Solves the PDE using Picard iterations
        
        Inputs:
        iterTol: Iterations stop when |u_(k) - u_(k-1)| < iterTol. Default: 1e-5
        maxIter: Maximum number of iterations
        dispOutput(bool): display iteration differences and exact errors at each iteration
        writeData(True/False): write solution and convergence data to files
        filePath: Path AND name of files WITHOUT file extension
        
        Outputs:
        u: solution to PDE
        iterDiffArray: Differences between iterative solutions (in L2 norm) at each iteration
        exactErrArray: Exact errors (in L2 norm) at each iteration
        
        Saved data:
        FEniCS solution saved to <filePath>.pvd
        Convergence data saved to <filePath>.csv:
            column 0: iterate differences
            column 1: exact errors
        """
        
        V = self.V
        u = TrialFunction(V)
        v = TestFunction(V)
        u_k = interpolate(Constant(0.0), V)  # previous (known) u
        a = inner(self.q(u_k)*grad(u), grad(v))*dx
        f = Constant(0.0)
        L = f*v*dx
        
        bcs = [self.B1, self.B2] 
        
        u = Function(V)     # new unknown function
        itErr = 1.0           # error measure ||u-u_k||
        iterDiffArray = []
        exactErrArray = []   
        iter = 0
        
        # Begin Picard iterations
        while itErr > iterTol and iter < maxIter:
            iter += 1
            
            solve(a == L, u, bcs)
            
            # calculate iterate difference and exact error in L2 norm
            itErr = errornorm(u_k, u, 'L2')
            exErr = errornorm(self.uExpr, u, 'L2')
            
            iterDiffArray.append(itErr) # fill arrays with error data
            exactErrArray.append(exErr)    
            
            if dispOutput:
                print('k = ' + str(iter) + ' | u-diff =  ' + str(itErr) + ', exact error = ' + str(exErr))
            u_k.assign(u)   # update for next iteration
        
        if writeData:
            # save solution
            solution = File(filePath + '.pvd')
            solution << u_k
            # save convergence data
            convergenceData = [iterDiffArray, exactErrArray]
            np.savetxt(filePath + '.csv', convergenceData)
            
        # save data to object
        self.picardSol = u_k
        self.picardIterDiff = iterDiffArray
        self.picardExactErr = exactErrArray
        return [u_k, iterDiffArray, exactErrArray]
        
    def solveNewton(self, iterTol = 1.0e-5, maxIter = 25, dispOutput = False, writeData = True, filePath = 'solution-data/PDE1Newton'):
        """
        Solves the PDE using Newton iterations
        
        Inputs:
        iterTol: Iterations stop when |u_(k) - u_(k-1)| < iterTol. Default: 1e-5
        maxIter: Maximum number of iterations
        dispOutput(True/False): display iteration differences and exact errors at each iteration
        writeData(True/False): write solution and convergence data to files
        filePath: Path AND name of files WITHOUT file extension
        
        Outputs:
        u: solution to PDE
        iterDiffArray: Differences between iterative solutions (in L2 norm) at each iteration
        exactErrArray: Exact errors (in L2 norm) at each iteration
        """
        
        V = self.V
        v = TestFunction(V)
        
        # construct initial guess (solution to PDE with q(u) = 1)
        u_k = TrialFunction(V)
        a0 = inner(grad(u_k), grad(v))*dx
        f = Constant(0.0)
        L0 = f*v*dx
        
        u_k = Function(V)
        bcs = [self.B1, self.B2]
        solve(a0 == L0, u_k, bcs)
        
        # construct problem in du          
        du = TrialFunction(V) # newton step
        a = (inner(self.q(u_k)*grad(du),grad(v)) + inner(self.dqdu(u_k)*du*grad(u_k),grad(v)))*dx     
        L = -inner(self.q(u_k)*grad(u_k),grad(v))*dx
        # du = 0 on boundaries
        B2 = DirichletBC(self.V, Constant(0.0), lambda x, on_boundary : self.right_boundary(x, on_boundary))
        bcs = [self.B1, B2]
        
        du = Function(V)
        u = Function(V)
        itErr = 1.0
        iterDiffArray = []
        exactErrArray = []
        iter = 0
        while itErr > iterTol and iter < maxIter:
            iter += 1
            
            solve(a == L, du, bcs)
            u.vector()[:] = u_k.vector() + du.vector()
            
            # calculate iterate difference and exact error in L2 norm
            itErr = errornorm(u_k, u, 'L2')
            exErr = errornorm(self.uExpr, u, 'L2')
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
        
    def plotConvergence(self):
        """
        Plots the convergence data (exact errors and iterate differences) for 
        Newton and/or Picard soltutions of the given PDE
        """
        # check which methods have been used to solve PDE           
        if hasattr(self, 'newtonSol'):
            plt.figure(1)
            plt.suptitle('Convergence data for PDE solution')
            
            plt.subplot(1,2,1)
            plt.plot(self.newtonExactErr)
            plt.ylabel('Newton exact error')
            plt.xlabel('iteration')
            plt.subplot(1,2,2)
            
            plt.plot(self.newtonIterDiff)
            plt.ylabel('Newton iterate difference')
            plt.xlabel('iteration')
        else:
            print 'No Newton solution calculated, run solveNewton method first'     
        if hasattr(self, 'picardSol'):
            plt.figure(2)
            plt.suptitle('Convergence data for PDE solution')
            
            plt.subplot(1,2,1)
            plt.plot(self.picardExactErr)
            plt.ylabel('Picard exact error')
            plt.xlabel('iteration')
            
            plt.subplot(1,2,2)
            plt.plot(self.picardIterDiff)
            plt.ylabel('Picard iterate difference')
            plt.xlabel('iteration')
        else:
            print 'No Picard solution calculated, run solvePicard method first'       
