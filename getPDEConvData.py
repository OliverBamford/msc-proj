from fenics import *
import numpy as np

def getPDEConvData(a, L, u_k, uexact, V, bcs = False, dispOutput = False):
    """
    Solves the PDE with weak form a==L with (optional) bcs on unit interval
    and exact solution uexact (passed as a function).
    Returns the data: [u_k, iterateNorm, exactErrNorm].
    dim: dimension of space (1 or 2)
    """
               
    u = Function(V)     # new unknown function
    itErr = 1.0           # error measure ||u-u_k||
    iterDiffArray = []
    exactErrArray = []   
        
    tol = 1.0E-5        # tolerance  <- should be an input
    iter = 0            # iteration counter
    maxiter = 25        # max no of iterations allowed
    
    # Begin Picard iterations
    while itErr > tol and iter < maxiter:
        iter += 1
        
        if bcs == False:
            solve(a == L, u)
        else:
            solve(a == L, u, bcs)
        
        # calculate iterate difference and exact error in L2 norm
        itErr = errornorm(u, u_k, 'L2')
        exErr = errornorm(u, uexact, 'L2')
        
        iterDiffArray.append(itErr) # fill array with error data
        exactErrArray.append(exErr)    
        
        if dispOutput:
            print('k = ' + str(iter) + ' | u-diff =  ' + str(itErr) + ', exact error = ' + str(exErr))
        u_k.assign(u)   # update for next iteration
    
    return [u_k, iterDiffArray, exactErrArray]
