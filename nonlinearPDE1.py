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
        BCtol = 1E-14
        def left_boundary(x, on_boundary):
            return on_boundary and abs(x[0]) < BCtol
        def right_boundary(x, on_boundary):
            return on_boundary and abs(x[0]-1) < BCtol
        B1 = DirichletBC(self.V, Constant(0.0), left_boundary) # u(0) = 0
        B2 = DirichletBC(self.V, Constant(1.0), right_boundary) # u(1) = 1
        self.bcs = [B1, B2]

        # construct exact answer in C format
        uExpr = Expression('pow((pow(2,m+1) - 1)*x[0] + 1,(1/(m+1))) - 1', m = 2, degree=3)
        # interpolate over function space
        self.ue = interpolate(uExpr, self.V)
        
    def q(self, u):
            return (1+u)**2
            
    def solvePicard(self, iterTol = 1.0e-5, maxIter = 25, dispOutput = False):
        """
        Solves the PDE using Picard iterations
        
        Inputs:
        iterTol: Iterations stop when |u_(k) - u_(k-1)| < iterTol. Default: 1e-5
        maxIter: Maximum number of iterations
        dispOutput(True/False): display iteration differences and exact errors at each iteration
        
        Outputs:
        u: solution to PDE
        iterDiffArray: Differences between iterative solutions (in L2 norm) at each iteration
        exactErrArray: Exact errors (in L2 norm) at each iteration
        """
        
        V = self.V
        u = TrialFunction(V)
        v = TestFunction(V)
        u_k = interpolate(Constant(0.0), V)  # previous (known) u
        a = inner(self.q(u_k)*grad(u), grad(v))*dx
        f = Constant(0.0)
        L = f*v*dx
        
        u = Function(V)     # new unknown function
        uexact = self.ue
        itErr = 1.0           # error measure ||u-u_k||
        iterDiffArray = []
        exactErrArray = []   
        iter = 0
        
        # Begin Picard iterations
        while itErr > iterTol and iter < maxIter:
            iter += 1
            
            solve(a == L, u, self.bcs)
            
            # calculate iterate difference and exact error in L2 norm
            itErr = errornorm(u, u_k, 'L2')
            exErr = errornorm(u, uexact, 'L2')
            
            iterDiffArray.append(itErr) # fill array with error data
            exactErrArray.append(exErr)    
            
            if dispOutput:
                print('k = ' + str(iter) + ' | u-diff =  ' + str(itErr) + ', exact error = ' + str(exErr))
            u_k.assign(u)   # update for next iteration
            
        return [u_k, iterDiffArray, exactErrArray]
        
    def solveNewton(self):
        