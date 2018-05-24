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
        self.alpha = 1e-07

        mesh = UnitSquareMesh(N,N)        
        U = FunctionSpace(mesh, 'CG', p)
        LMBD = FunctionSpace(mesh, 'CG', p)
        M = FunctionSpace(mesh, 'CG', p)
        u = Function(U)
        lmbd = Function(LMBD)
        m = Function(M)
        
        self.bcs = [DirichletBC(U, 0, "on_boundary"),
               DirichletBC(LMBD, 0, "on_boundary")]
        
        dist = Expression('sin(15*x[0]) + cos(20*x[1])', degree=3)
        # ud = interpolate(dist, Z.sub(0).collapse())
        self.ud = interpolate(dist, U)
        
        self.u_k = u
        self.lmbd_k = lmbd
        self.m_k = m      
        self.U = U
        self.LMBD = LMBD
        self.M = M
        
    def solveAuto(self):
        """
        Solves problem using FEniCS automatic solver (BROKEN)
        """
        u = self.u_k
        ud = self.ud
        m = self.m_k
        lmbd = self.lmbd_k
        
        self.L = (0.5*inner(u-ud, u-ud)*dx
                + 0.5*self.alpha*inner(m, m)*dx
                + inner(grad(u), grad(lmbd))*dx
                - m*lmbd*dx)
        
        
        self.F = derivative(self.L, self.z, TestFunction(self.Z))
        solve(self.F == 0, self.z, self.bcs)
        
    def solveSD(self, srch = 1, iterTol = 1.0e-5, maxIter = 25, dispOutput = False, writeData = True, filePath = 'solution-data/PDEOptSD'):
        u_k = self.u_k
        lmbd_k = self.lmbd_k
        m_k = self.m_k
        ud = self.ud
        U = self.U
        LMBD = self.LMBD
        M = self.M
        
        # find Riesz-rep of dJ (GJ)
        itErr = 1.0           # error measure ||u-u_k||
        iterDiffArray = []
        exactErrArray = []   
        iter = 0
        
        u_k = ud # initial guesses
        lmbd = interpolate(Constant(1.0), LMBD)
        m_k = interpolate(Constant(1.0), M) 
        
        # begin steepest descent
        while itErr > iterTol and iter < maxIter:
            iter += 1
            
            # find the Riesz rep. of dJ 
            GJ = TrialFunction(M)
            v = TestFunction(M)
            a = GJ*v*dx
            L = (self.alpha*m_k - lmbd_k)*v*dx
            GJ = Function(M)
            solve(a == L, GJ)
            # update m
            m_k.assign(m_k - srch*GJ)
            
            # update u
            u = TrialFunction(U)
            v = TestFunction(U)
            State = inner(grad(v),grad(u))*dx
            L = -m_k*v*dx
            u = Function(U)
            solve(State == L, u, self.bcs[0])
            u_k.assign(u)
            
            # update lambda
            lmbd = TrialFunction(LMBD)
            v = TestFunction(LMBD)
            Adj = inner(grad(v),grad(lmbd))*v*dx
            L = -(u_k-ud)*v*dx
            lmbd = Function(LMBD)
            solve(Adj == L, lmbd, self.bcs[1])
            lmbd_k.assign(lmbd)
            
            
    