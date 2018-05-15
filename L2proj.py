from fenics import *
import matplotlib.pyplot as plt
import numpy as np

class L2proj:
    def __init__(self, N, p, d, f):
        """
        Sets up the L2 projection problem:
        min ||u - f||
        over a unit line or unit square space
        
        
        Inputs:
        N: number of finite elements in mesh
        p: order of function space
        d: dimension of function space (1 or 2)
        f: UFL expression
        """
        self.f = f
        
        # set up function space
        if d == 1:
            mesh = UnitIntervalMesh(N)
        elif d == 2:
            mesh = UnitSquareMesh(N,N)
        else:
            #experiments in higher dimensions are too expensive
            raise NotImplementedError
        self.V = FunctionSpace(mesh, 'CG', p)
                            
    def solveSD(self, alpha = 1, iterTol = 1.0e-5, maxIter = 25, dispOutput = False, writeData = True, filePath = 'solution-data/L2projSD'):
        """
        Finds the L2 projection of f using steepest descent
        
        Inputs:
        alpha: SD step size
        iterTol: Iterations stop when |u_(k) - u_(k-1)| < iterTol. Default: 1e-5
        maxIter: Maximum number of iterations
        dispOutput(True/False): display iteration differences and exact errors at each iteration
        writeData(True/False): write solution and convergence data to files
        filePath: Path AND name of files WITHOUT file extension
        
        Outputs:
        u: L2 projection of f
        iterDiffArray: Differences between iterative solutions (in L2 norm) at each iteration
        exactErrArray: Exact errors (in L2 norm) at each iteration
        """
        V = self.V
        v = TestFunction(V)
        u = Function(V)     # new unknown function
        itErr = 1.0           # error measure ||u-u_k||
        iterDiffArray = []
        exactErrArray = []   
        iter = 0
        
        f = self.f
        u_k = interpolate(Constant(0.0), V) #initial guess u_0
        # begin steepest descent
        while itErr > iterTol and iter < maxIter:
            iter += 1
            
            # find grad(F) using current u iterate
            GF = TrialFunction(V)
            a = GF*v*dx
            L = (u_k - f)*v*dx
            GF = Function(V)
            solve(a == L, GF)
            
            u.assign(u_k - alpha * GF) # calculate errors
            itErr = errornorm(u_k, u, 'L2')
            exErr = errornorm(f, u, 'L2')
            
            iterDiffArray.append(itErr) # fill arrays with error data
            exactErrArray.append(exErr)    
        
            if dispOutput:
                print('k = ' + str(iter) + ' | u-diff =  ' + str(itErr) + ', exact error = ' + str(exErr))
                
            # update u iterate 
            u_k.assign(u)
        
        if writeData:
            # save solution
            solution = File(filePath + '.pvd')
            solution << u_k
            # save convergence data
            convergenceData = [iterDiffArray, exactErrArray]
            np.savetxt(filePath + '.csv', convergenceData)
            
        # save data to object
        self.SDSol = u_k
        self.SDIterDiff = iterDiffArray
        self.SDExactErr = exactErrArray
        return [u_k, iterDiffArray, exactErrArray]
        
    def solveNewton(self, iterTol = 1.0e-5, maxIter = 25, dispOutput = False, writeData = True, filePath = 'solution-data/L2projNewton'):
        """
        Finds the L2 projection of f using Newton iterations
        
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
        f = self.f

        u_k = interpolate(Constant(0.0), V) #initial guess u_0
        
        # construct problem in Newton step du
        du = TrialFunction(V)
        a = du*v*dx
        L = -(u_k - f)*v*dx        
        
                
        du = Function(V)
        u = Function(V)
        itErr = 1.0
        iterDiffArray = []
        exactErrArray = []
        iter = 0
        while itErr > iterTol and iter < maxIter:
            iter += 1
            
            solve(a == L, du)
            u.vector()[:] = u_k.vector() + du.vector()
            
            # calculate iterate difference and exact error in L2 norm
            itErr = errornorm(u_k, u, 'L2')
            exErr = errornorm(f, u, 'L2')
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
        return [u_k, iterDiffArray, exactErrArray]
        
    def plotConvergence(self):
        """
        Plots the convergence data (exact errors and iterate differences) for 
        Newton and/or SD soltutions of the given L2 projection
        """
        
        # check which methods have been used to find L2 projection
        if hasattr(self, 'newtonSol'):
            plt.figure(1)
            plt.suptitle('Convergence data for L2 projection')
            
            plt.subplot(1,2,1)
            plt.plot(self.newtonExactErr)
            plt.ylabel('Newton exact error')
            plt.xlabel('iteration')
            
            plt.subplot(1,2,2)
            plt.plot(self.newtonIterDiff)
            plt.ylabel('Newton iterate difference')
            plt.xlabel('iteration')
            
            if hasattr(self, 'SDSol'):
                plt.figure(2)
                plt.suptitle('Convergence data for L2 projection')
                
                plt.subplot(1,2,1)
                plt.plot(self.SDExactErr)
                plt.ylabel('SD exact error')
                plt.xlabel('iteration')
                
                plt.subplot(1,2,2)
                plt.plot(self.SDIterDiff)
                plt.ylabel('SD iterate difference')
                plt.xlabel('iteration')
            else:
                print 'No SD solution calculated, run solveSD method first'   
                
        elif hasattr(self, 'SDSol'):
                print 'No Newton solution calculated, run solveNewton method first'
                plt.figure(1)
                plt.suptitle('Convergence data for L2 projection')
                
                plt.subplot(1,2,1)
                plt.plot(self.SDExactErr)
                plt.ylabel('SD exact error')
                plt.xlabel('iteration')
                
                plt.subplot(1,2,2)
                plt.plot(self.SDIterDiff)
                plt.ylabel('SD iterate difference')
                plt.xlabel('iteration')
        else:
            print 'Nothing to plot...'
            


            