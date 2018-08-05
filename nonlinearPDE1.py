from fenics import *
import matplotlib.pyplot as plt
import numpy as np

class nonlinearPDE1:
    def __init__(self, N, p):
        """
        Sets up the 2D PDE: -grad((1+u)^2.grad(u)) = 0
        
        Inputs:
        N: number of finite elements in mesh
        p: order of function space
        """
        # set up function space
        self.d = d
        self.mesh = UnitSquareMesh(N,N)
        self.V = FunctionSpace(self.mesh, 'CG', p)
        v = TestFunction(V)
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
                    writeData = True, ernErrors = False,
                    filePath = 'solution-data/PDE1Picard'):
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
        L0 = f*v*dx
        u_k = Function(V)
        solve(a0 == L0, u_k, bcs)
        a = inner(self.q(u_k)*grad(u), grad(v))*dx
        L = self.f*v*dx
        
        
        
        u = Function(V)     # new unknown function
        itErr = 1.0           # error measure ||u-u_k||
        iterDiffArray = [errornorm(u_k, u, 'H1')]
        exactErrArray = [errornorm(self.uExpr, u, 'H1')]
        iter = 0
        if ernErrors: error_estimators = np.array([[0.,0.,0.,0.]])
        # Begin Picard iterations
        while itErr > iterTol and iter < maxIter:
            iter += 1
            
            solve(a == L, u, bcs)
            
            # calculate iterate difference and exact error in L2 norm
            itErr = errornorm(u_k, u, 'H1')
            exErr = errornorm(self.uExpr, u, 'H1')
            
            if ernErrors:
                np.concatenate((error_estimators, np.array([self.calculateErnErrors(u, u_k)])), axis = 0)
            
            iterDiffArray.append(itErr) # fill arrays with error data
            exactErrArray.append(exErr)
            
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
        iterDiffArray = [errornorm(u_k, u, 'H1')]
        exactErrArray = [errornorm(self.uExpr, u, 'H1')]
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
    
    def calculateErnErrors(self, u, u_k):
        gu = project(grad(u), self.F) #TODO: make sure this is legit
        self.sigmakBar = interpolate(Expression(['(1 + u)*(1 + u)*gu[0]', 
                                        '(1 + u)*(1 + u)*gu[1]'], 
                                            u = u_k, gu = gu, degree=0), self.F0)
                                            # u = u_k (previous u iterate)
        self.sigmakBar = interpolate(self.sigmakBar, self.RTN)
        self.sigmaBar = interpolate(Expression(['(1 + u)*(1 + u)*gu[0]', 
                                            '(1 + u)*(1 + u)*gu[1]'], 
                                                u = u, gu = gu, degree=0), self.F0)
        self.sigmaBar = interpolate(self.sigmaBar, self.RTN)
        
        # construct sum (second terms Eqns (6.7) and (6.9) for each cell K
        # find residual for each edge using 'test function trick'
        R_eps = assemble(self.R_eps_form)
        Rbar_eps = assemble(self.Rbar_eps_form)
        rk = Function(self.F)
        r = Function(self.F)
        rk_ = np.zeros(rk.vector().get_local().shape)
        r_ = np.zeros(r.vector().get_local().shape)
        eta_disc = 0.
        eta_quad = 0.
        eta_osc = 0.
        for cell in cells(self.mesh):
            dofs = self.dm.cell_dofs(cell.index()) # get indices of dofs belonging to cell
            rk_c = rk.vector().get_local()[dofs]
            r_c = r.vector().get_local()[dofs]
            x_c = self.x_.vector().get_local()[dofs]
            myEdges = edges(cell)
            myVerts = vertices(cell)
            eps_K = [myEdges.next(), myEdges.next(), myEdges.next()]
            a_K = [myVerts.next(), myVerts.next(), myVerts.next()]
            # |T_e| = 2 for all internal edges
            cardT_e = 2.
            eta_NCK = 0.
            # a_K[n] is the vertex opposite to the edge eps_K[n]
            for i in range(0, len(eps_K)-1):
               cardT_e = eps_K[i].entities(self.d).size # number of cells sharing e 
               R_e = R_eps[eps_K[i].index()][0] # find residual corresponding to edge
               Rbar_e = Rbar_eps[eps_K[i].index()][0] # find barred residual corresponding to edge          
               # find distance between all dofs on cell and vertex opposite to edge
               rk_c[0:rk_c.size/2] += 1./(cardT_e*self.d)*R_e*(x_c[0:rk_c.size/2] - a_K[i].point().array()[0]) 
               rk_c[rk_c.size/2:rk_c.size] += 1./(cardT_e*self.d)*R_e*(x_c[rk_c.size/2:rk_c.size] - a_K[i].point().array()[1]) 
               r_c[0:r_c.size/2] += 1./(cardT_e*self.d)*Rbar_e*(x_c[0:r_c.size/2] - a_K[i].point().array()[0]) 
               r_c[r_c.size/2:r_c.size] += 1./(cardT_e*self.d)*Rbar_e*(x_c[r_c.size/2:r_c.size] - a_K[i].point().array()[1]) 
               # s := q = 2
               if eps_K[i].entities(d).size > 1: # if edge is internal (else jump(u) should = 0)
                   self.mf.set_value(eps_K[i].mesh_id(), 1) # mark domain to integrate over
                   eta_NCK += assemble(jump(u)*jump(u)*dS(subdomain_data=self.mf)) / eps_K[i].length() # squared L2 norm of jump along edge
                   self.mf.set_value(eps_K[i].mesh_id(), 0) # un-mark domain
            # add squared local discretisation estimator to total
            eta_disc += 2**(0.5)*(assemble_local((dflux+sigmaBar)**2*dx, cell)**(0.5) + eta_NCK)**2
            eta_osc += cell.h()/np.pi * assemble_local((f - self.f_h)**2*dx, cell)**(0.5)
            rk_[dofs] = rk_c # place cell values back into main array
            r_[dofs] = r_c # place cell values back into main array
            
        rk.vector().set_local(rk_)
        r.vector().set_local(r_)
        
        # interpolate CR construction of residuals into RTN
        rk = interpolate(rk, self.RTN)
        r = interpolate(r, self.RTN)    
        # compute global discretisation and quadrature estimators
        eta_disc = eta_disc**(0.5)
        eta_quad = assemble((q(u)*gu-sigmaBar)**2*dx)**(0.5)
        # construct flux (d+l) for each element (Eq. (6.7))
        dflux.assign(-sigmaBar + self.f_hvec - r)
        lflux.assign(-sigmakBar + self.f_hvec - rk - self.dflux)
        eta_lin = norm(self.lflux, 'L2')**(0.5)
        
        return np.array([eta_disc, eta_lin, eta_quad, eta_osc])

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
            plt.figure(1)
            plt.semilogy(self.newtonExactErr, 'r^-', linewidth=2, markersize=10)
            plt.semilogy(self.newtonIterDiff, 'b^-', linewidth=2, markersize=10)
            plt.ylabel('$H^1$ error', fontsize=40)
            plt.xlabel('$k$', fontsize=40)
            plt.legend(['Exact error', 'Iterate difference'], loc=3, fontsize=30)
            plt.tick_params(labelsize=25)
        else:
            print 'No Newton solution calculated, run solveNewton method first'     
        if hasattr(self, 'picardSol'):
            plt.figure(2)
            plt.semilogy(self.picardExactErr, 'r^-', linewidth=2, markersize=10)
            plt.semilogy(self.picardIterDiff, 'b^-', linewidth=2, markersize=10)
            plt.ylabel('$H^1$ error', fontsize=40)
            plt.xlabel('$k$', fontsize=40)
            plt.legend(['Exact error', 'Iterate difference'], loc=3, fontsize=30)
            plt.tick_params(labelsize=25)
        else:
            print 'No Picard solution calculated, run solvePicard method first'       