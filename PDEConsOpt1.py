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
        
        bcs = [DirichletBC(Z.sub(0), 0, "on_boundary"),
               DirichletBC(Z.sub(1), 0, "on_boundary")]
        
        dist = Expression('sin(15*x[0]) + cos(20*x[1])', degree=3)
        ud = interpolate(dist, Z.sub(0).collapse)
        
        L = (0.5*inner(u-ud, u-ud)*dx
            + 0.5*alpha*inner(m, m)*dx
            + inner(grad(u), grad(lmbda))*dx
            - m*lmbda*dx)
        
        F = derivative(L, z, TestFunction(Z))
         
        solve(F == 0, z, bcs)