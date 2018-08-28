from fenics import *
import matplotlib.pyplot as plt
import numpy as np

class nonlinearPDE1:
    def __init__(self, N):
        """
        Sets up the 2D PDE: -grad((1+u)^2.grad(u)) = 0
        
        Inputs:
        N: number of finite elements in mesh
        """
        # set up function space
        self.mesh = UnitSquareMesh(N,N)
        self.V = FunctionSpace(self.mesh, 'CR', 1)
        v = TestFunction(self.V)
        # set up BCs on left and right
        # lambda functions ensure the boundary methods take two variables
        self.B1 = DirichletBC(self.V, Constant(0.0), lambda x, on_boundary : self.left_boundary(x, on_boundary)) # u(0) = 0
        self.B2 = DirichletBC(self.V, Constant(1.0), lambda x, on_boundary : self.right_boundary(x, on_boundary)) # u(1) = 1

        # construct exact solution in C format
        self.uExpr = Expression('pow((pow(2,m+1) - 1)*x[0] + 1,(1/(m+1))) - 1', m = 2, degree=4)
        self.f = Constant(0.0)
        
        
    def left_boundary(self, x, on_boundary):
            return on_boundary and abs(x[0]) < 1E-14
    def right_boundary(self, x, on_boundary):
            return on_boundary and abs(x[0]-1) < 1E-14
    def q(self, u):
            return (1+u)**2
    def dqdu(self,u):
            return 2*(1+u)
            
    def solvePicard(self, iterTol = 1.0e-5, maxIter = 25, dispOutput = False, 
                    writeData = True, filePath = 'solution-data/PDE1Picard'):
        """
        Solves the PDE using Picard iterations
        
        Inputs:
        iterTol: Iterations stop when |u_(k) - u_(k-1)| < iterTol. Default: 1e-5
        maxIter: Maximum number of iterations
        dispOutput(bool): display iteration differences and exact errors at each iteration
        writeData(True/False): write solution and convergence data to files
        ernErrors(True/False): calculate Ern error estimated at each iteration
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
        
        V = self.V
        bcs = [self.B1, self.B2] 
        u = TrialFunction(V)
        v = TestFunction(V)
        u_k = TrialFunction(V)
        a0 = inner(grad(u_k), grad(v))*dx
        L0 = self.f*v*dx
        u_k = Function(V)
        solve(a0 == L0, u_k, bcs)
        a = inner(self.q(u_k)*grad(u), grad(v))*dx
        L = self.f*v*dx

        u = Function(V)     # new unknown function
        itErr = 1.0           # error measure ||u-u_k||
        iterDiffArray = [errornorm(u_k, u, 'H1')]
        exactErrArray = [errornorm(self.uExpr, u, 'H1')]
        R = assemble(self.q(u)*inner(grad(u),grad(v))*dx)
        self.B1.apply(R, u.vector())
        self.B2.apply(R, u.vector())
        residualArr = [R.norm('l2')]
        iter = 0
        # Begin Picard iterations
        while itErr > iterTol and iter < maxIter:
            iter += 1
            
            solve(a == L, u, bcs)
            
            # calculate iterate difference and exact error in L2 norm
            itErr = errornorm(u_k, u, 'H1')
            exErr = errornorm(self.uExpr, u, 'H1')
            iterDiffArray.append(itErr) # fill arrays with error data
            exactErrArray.append(exErr)
            R = assemble(self.q(u)*inner(grad(u),grad(v))*dx)
            self.B1.apply(R, u.vector())
            self.B2.apply(R, u.vector())
            residualArr.append(R.norm('l2'))
            
            if dispOutput:
                print('k = ' + str(iter) + ' | u-diff =  ' + str(itErr) + ', exact error = ' + str(exErr))
            u_k.assign(u) # update for next iteration
        
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
        self.picardResArr = residualArr
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
        iterDiffArray: Differences between iterative solutions (in H1 norm) at each iteration
        exactErrArray: Exact errors (in H1 norm) at each iteration
        
        Saved data:
        FEniCS solution saved to <filePath>.pvd
        Convergence data saved to <filePath>.csv:
            column 0: iterate differences
            column 1: exact errors
        """
        
        V = self.V
        v = TestFunction(V)
        
        # construct initial guess (solution to PDE with q(u) = 1)
        u_k = TrialFunction(V)
        a0 = inner(grad(u_k), grad(v))*dx
        f = Constant(0.0)
        L0 = self.f*v*dx
        
        u_k = Function(V)
        bcs = [self.B1, self.B2]
        solve(a0 == L0, u_k, bcs)
        
        # construct problem in du          
        du = TrialFunction(V) # newton step
        a = (inner(self.q(u_k)*grad(du),grad(v)) + inner(self.dqdu(u_k)*du*grad(u_k),grad(v)))*dx     
        L = -inner(self.q(u_k)*grad(u_k),grad(v))*dx
        # du = 0 on boundaries
        B2_du = DirichletBC(self.V, Constant(0.0), lambda x, on_boundary : self.right_boundary(x, on_boundary))
        bcs = [self.B1, B2_du]
        
        du = Function(V)
        u = Function(V)
        itErr = 1.0
        iterDiffArray = [errornorm(u_k, u, 'H1')]
        exactErrArray = [errornorm(self.uExpr, u, 'H1')]
        R = assemble(self.q(u)*inner(grad(u),grad(v))*dx)
        self.B1.apply(R, u.vector())
        self.B2.apply(R, u.vector())
        residualArr = [R.norm('l2')]
        iter = 0
        while itErr > iterTol and iter < maxIter:
            iter += 1
            
            solve(a == L, du, bcs)
            u.vector()[:] = u_k.vector() + du.vector()
            
            # calculate iterate difference and exact error in L2 norm
            itErr = errornorm(u_k, u, 'H1')
            exErr = errornorm(self.uExpr, u, 'H1')
            iterDiffArray.append(itErr) # fill arrays with error data
            exactErrArray.append(exErr)
            R = assemble(self.q(u)*inner(grad(u),grad(v))*dx)
            self.B1.apply(R, u.vector())
            self.B2.apply(R, u.vector())
            residualArr.append(R.norm('l2'))
            
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
        self.newtonResArr = residualArr
        return [u_k, iterDiffArray, exactErrArray]

    def plotConvergence(self):
        """
        Plots the convergence data (exact errors and iterate differences) for 
        Newton and/or Picard soltutions of the given PDE
        """
        from matplotlib import rc
        import matplotlib.pylab as plt
        
        rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        rc('text', usetex=True)
        # check which methods have been used to solve PDE           
        if hasattr(self, 'newtonSol'):
            plt.figure(1, figsize = (12,10))
            plt.semilogy(self.newtonExactErr, 'r^-', linewidth=2, markersize=10)
            plt.semilogy(self.newtonIterDiff, 'b^-', linewidth=2, markersize=10)
            plt.semilogy(self.newtonResArr, 'g^-', linewidth=2, markersize=10)
            plt.ylabel('Error', fontsize=40)
            plt.xlabel('Iteration', fontsize=40)
            #plt.legend(['Exact error', 'Iterate difference', 'Residual'], loc=3, fontsize=30)
            plt.tick_params(labelsize=25)
        else:
            print('No Newton solution calculated, run solveNewton method first')    
        if hasattr(self, 'picardSol'):
            plt.figure(2, figsize = (12,10))
            plt.semilogy(self.picardExactErr, 'r^-', linewidth=2, markersize=10)
            plt.semilogy(self.picardIterDiff, 'b^-', linewidth=2, markersize=10)
            plt.semilogy(self.picardResArr, 'g^-', linewidth=2, markersize=10)
            plt.ylabel('Error', fontsize=40)
            plt.xlabel('$k$', fontsize=40)
            plt.legend(['Exact error', 'Iterate difference', 'Residual'], loc=3, fontsize=30)
            plt.tick_params(labelsize=25)
        else:
            print('No Picard solution calculated, run solvePicard method first')

myPDE = nonlinearPDE1(30)
myPDE.solvePicard(iterTol = 0, maxIter = 10, dispOutput = True)
myPDE.solveNewton(iterTol = 0, maxIter = 10, dispOutput = True)
myPDE.plotConvergence()