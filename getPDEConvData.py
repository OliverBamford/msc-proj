from fenics import *
import numpy as np

def getPDEConvData(a, f, bcs, uexact, meshSize, deg, dispOutput = False):
    """
    Solves the 1D PDE with weak form a==f*v*dx with bcs on unit interval
    and exact solution uexact (passed as a function).
    Returns two arrays: [iterateNorm, exactErrNorm].
    """
    
    
    mesh = UnitIntervalMesh(meshSize)
    V = FunctionSpace(mesh, 'CG', deg)
    
    v = TestFunction(V)
    u_k = interpolate(Constant(0.0), V)  # previous (known) u
    L = f*v*dx
     
    u = Function(V)     # new unknown function
    err = 1.0           # error measure ||u-u_k||
    iterDiffArray = []
    exactErrArray = []
    x = np.linspace(1,0,deg*meshSize + 1)
    UE = uexact(x) # exact solution over unit interval
    
    tol = 1.0E-5        # tolerance
    iter = 0            # iteration counter
    maxiter = 25        # max no of iterations allowed
    
    while err > tol and iter < maxiter:
        iter += 1
        solve(a == L, u, bcs)
        
        diff = u.vector().get_local() - u_k.vector().get_local()
        err = np.linalg.norm(diff, ord=np.Inf)
        exDiff = u.vector().get_local() - UE # calculate exact error
        exErr = np.linalg.norm(exDiff, ord=np.Inf)
        
        iterDiffArray.append(err) # fill array with error data
        exactErrArray.append(exErr)    
        if dispOutput:
            print('k = ' + str(iter) + ' | u-diff =  ' + str(err) + ', exact error = ' + str(exErr))
        u_k.assign(u)   # update for next iteration
    
    return [iterDiffArray, exactErrArray]