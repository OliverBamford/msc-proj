from fenics import *
import numpy as np

def getPDEConvData(a, f, u_k, uexact, mesh, dim, fnSpace, meshSize, deg, bcs = False, dispOutput = False):
    """
    Solves the 1D PDE with weak form a==f*v*dx with (optional) bcs on unit interval
    and exact solution uexact (passed as a function).
    Returns two arrays: [iterateNorm, exactErrNorm].
    dim: dimension of space (1 or 2)
    """
    
    V = fnSpace  
    v = TestFunction(V)
    L = f*v*dx
    
    #sample solution, not necessary with errornorm.. update this
    if dim == 1:
        x = np.linspace(1,0,deg*meshSize + 1)
        UE = uexact(x) # exact solution over unit interval
    elif dim == 2:
        x = np.linspace(1,0,deg*meshSize + 1)
        y = np.linspace(1,0,deg*meshSize + 1)
        meshE = np.meshgrid(x,y)
        UE = uexact(meshE)
        
    u = Function(V)     # new unknown function
    err = 1.0           # error measure ||u-u_k||
    iterDiffArray = []
    exactErrArray = []   
        
    tol = 1.0E-5        # tolerance  <- should be an input
    iter = 0            # iteration counter
    maxiter = 25        # max no of iterations allowed
    
    # Begin Picard iterations
    while err > tol and iter < maxiter:
        iter += 1
        
        if bcs == False:
            solve(a == L, u)
        else:
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
