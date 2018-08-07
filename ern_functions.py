from fenics import *
import numpy as np

def get_fvec(f_h, mesh):
    """Calculation of f_h vector for Ern errors
        General Idea: Interpolate cartesian coords
        into DG space, which has pointwise evaluation
        dofs, then subtract cell midpoints from
        corresponding edges.
        Inputs:
        f_h: projection of f into DG0
    """
    d = mesh.geometry().dim()
    RT = VectorFunctionSpace(mesh, 'DG', 1, dim=d)
    X = MeshCoordinates(mesh)
    f_hvec = interpolate(X, RT)
    f_ = np.zeros(f_hvec.vector().get_local().shape) # create np array which contains values to be assigned to f_hvec
    dm = RT.dofmap()
    for cell in cells(mesh):
        dofs = dm.cell_dofs(cell.index())
        f_c = f_hvec.vector().get_local()[dofs] # get np array of all dof values in cell
        f_c *= f_h.vector().get_local()[cell.index()]/d
        mp = cell.midpoint().array()# get midpoint of cell
        f_c[0:f_c.size/2] -= f_h.vector().get_local()[cell.index()]*mp[0]/d # construct f_h in cell
        f_c[f_c.size/2:f_c.size] -= f_h.vector().get_local()[cell.index()]*mp[1]/d # note [x,x,x,y,y,y] structure of f_c
        f_[dofs] = f_c # place cell values back into main array
    f_hvec.vector().set_local(f_)
    return f_hvec
    
def get_estimators(V, f, f_h, f_hvec, sigma, sigma_lin, u, u_k, mesh, p):
    v = TestFunction(V)
    n = FacetNormal(mesh)
    qu = p/float(p-1)
    d = mesh.geometry().dim()
    RT = VectorFunctionSpace(mesh, 'DG', degree=1, dim=d)
    F0 = VectorFunctionSpace(mesh, 'DG', degree=0, dim=d) # space for 0-order interpolants (sigma)
    dm = RT.dofmap()
    dm0 = F0.dofmap()
    dmV = V.dofmap()
    edge_dofs = dmV.entity_dofs(mesh, 1) # get dofs associated with edge indexes
    sigmakBar0 = project(sigma_lin(u, u_k), F0)
    sigmaBar0 = project(sigma(u), F0)
    dflux = Function(RT)
    lflux = Function(RT)
    mf = MeshFunctionSizet(mesh, d-1, 0) # used to mark edges for integration
    x_ = interpolate(MeshCoordinates(mesh), RT) # used as 'x' vector when constructing flux
    # construct sum (second terms Eqns (6.7) and (6.9) for each cell K
    # find residual for each edge using 'test function trick'
    R_eps = assemble(f_h*v*dx - inner(sigma_lin(u, u_k), grad(v))*dx)
    Rbar_eps = assemble(f_h*v*dx - inner(sigma(u), grad(v))*dx)
    rk = Function(RT)
    r = Function(RT)
    rk_ = np.zeros(rk.vector().get_local().shape)
    r_ = np.zeros(r.vector().get_local().shape)
    eta_disc = 0.
    eta_quad = 0.
    eta_osc = 0.
    eta_NC = 0.
    eta_disc_noNC = 0.
    d_ = np.zeros(dflux.vector().get_local().shape)
    l_ = np.zeros(lflux.vector().get_local().shape)
    for cell in cells(mesh):
        dofs = dm.cell_dofs(cell.index()) # get indices of dofs belonging to cell
        dofs0 = dm0.cell_dofs(cell.index()) # and for the 0th order projections
        rk_c = rk.vector().get_local()[dofs]
        r_c = r.vector().get_local()[dofs]
        dflux_c = dflux.vector().get_local()[dofs]
        lflux_c = lflux.vector().get_local()[dofs]
        x_c = x_.vector().get_local()[dofs]
        myEdges = edges(cell)
        myVerts = vertices(cell)
        eps_K = [myEdges.next(), myEdges.next(), myEdges.next()]
        a_K = [myVerts.next(), myVerts.next(), myVerts.next()]
        mp = cell.midpoint().array()[0:2] # midpoint is returned in (x,y,z) format
        eta_NCK = 0.
        # a_K[n] is the vertex opposite to the edge eps_K[n]
        for i in range(len(eps_K)):
           cardT_e = eps_K[i].entities(d).size # number of cells sharing e
           R_e = R_eps[edge_dofs[eps_K[i].index()]][0] # find residual corresponding to edge
           Rbar_e = Rbar_eps[edge_dofs[eps_K[i].index()]][0] # find barred residual corresponding to edge          
           # find distance between all dofs on cell and vertex opposite to edge
           rk_c += 1./(cardT_e*d)*R_e*(x_c - mp.repeat(3))
           r_c += 1./(cardT_e*d)*Rbar_e*(x_c - mp.repeat(3))
           # s = q
           mf.set_value(eps_K[i].index(), 1) # mark domain to integrate over
           eta_NCK += assemble(inner(jump(u,n), jump(u,n))**(qu/2)*dS(1, subdomain_data=mf)) / eps_K[i].length() # Lq-norm of jump along edge^q
           mf.set_value(eps_K[i].index(), 0) # un-mark domain
        
        dflux_c = -sigmaBar0.vector().get_local()[dofs0].repeat(3) + f_hvec.vector().get_local()[dofs] - r_c
        lflux_c = -sigmakBar0.vector().get_local()[dofs0].repeat(3) + f_hvec.vector().get_local()[dofs] - rk_c - dflux_c
        d_[dofs] = dflux_c
        l_[dofs] = lflux_c
        eta_NCK = eta_NCK**(1/qu)
        dflux.vector().set_local(d_)
        # add local discretisation estimator^q to total
        eta_disc += (2**(1./p)*(assemble_local((inner(dflux+sigmaBar0,dflux+sigmaBar0))**(qu/2)*dx, cell)**(1/qu) + eta_NCK))**qu
        eta_osc += cell.h()/np.pi*assemble_local((inner(f - f_h, f - f_h))**(qu/2)*dx, cell)
        eta_NC += eta_NCK
        rk_[dofs] = rk_c # place cell values back into main array
        r_[dofs] = r_c # place cell values back into main array
    
    lflux.vector().set_local(l_)
    rk.vector().set_local(rk_)
    r.vector().set_local(r_)
    # compute global discretisation and quadrature estimators
    eta_disc = eta_disc**(1/qu)
    eta_NC = eta_NC**(1/qu)
    eta_quad = assemble((inner(sigma(u)-sigmaBar0,sigma(u)-sigmaBar0))**(qu/2)*dx)**(1/qu)
    eta_osc = eta_osc**(1/qu)
    # construct flux (d+l) for each element (Eq. (6.7))
    #dflux.assign(-sigmaBar + f_hvec - r) 
    #lflux.assign(-sigmakBar + f_hvec - rk - dflux)
    #lflux = interpolate(lflux_vec, RTN)
    eta_lin = assemble(inner(lflux,lflux)**(qu/2)*dx)**(1/(2*qu))
    #print(project(div(lflux + dflux), RT).vector().get_local())
    return eta_disc, eta_lin, eta_quad, eta_osc, eta_NC

def solve_2D_flux_PDE(q, f, N, p, bcs, dqdu = None, 
                      dqdg = None, u0 = None, exact_solution = None, 
                      solver = 'Newton'):
    """
    Solves the 2D flux-type PDE:
    find u in L^p((0,1)x(0,1)) such that:
    -div(q(u, grad(u))*grad(u)) = f
    + Dirichlet BCs
    using linear Crouzeix-Raviart finite elements
    
    Inputs:
    q: function handle
    dqdu: function handle, partial derivative dq/du
    dqdg: function handle, partial derivative dqd(grad(u))
    f: FEniCS function
    N: Mesh refinement
    p: integer
    bcs: FEniCS Dirichlet BCs (or a tuple of Dirichlet BCs)
    u0: UFL expression, initial guess. If not specified, solution to PDE with
    q(u) = 1 will be used.
    exact_solution: UFL expression for analytic solution of PDE, if applicable
    solver: string, 'Newton' or 'Picard'
    
    Outputs:
    u: FEniCS function that solves the PDE
    error_estimators: Ern error estimators from each iteration in format
    [eta_disc, eta_lin, eta_quad, eta_osc, eta_NC]
    """
    
    ### MESH SETUP ###
    mesh = UnitSquareMesh(N,N)
    mesh.init()
    V = FunctionSpace(mesh, 'CR', 1)
    if exact_solution != None:
        u_e = interpolate(exact_solution, V)
    f_h = interpolate(f, FunctionSpace(mesh, 'DG', 0))
    v = TestFunction(V)
    
    ### INITIAL GUESS ###
    if u0 == None:
        u_k = TrialFunction(V)
        a0 = inner(grad(u_k), grad(v))*dx
        L0 = f_h*v*dx
        u_k = Function(V)
        solve(a0 == L0, u_k, bcs)
    else: u_k = interpolate(u0, V)
    
    ### PROBLEM SET UP ###
    u = Function(V)
    u.assign(u_k) # load up initial guess
    sigma = lambda u: q(u)*grad(u) # nonlinear flux function
    qu = p/float(p-1)
    if solver == 'Newton':  
        # make sure these are callable
        if dqdu == None:
            dqdu = lambda u : 0*u
        if dqdg == None:
            dqdg = lambda u : 0*u    
        bcs_du = DirichletBC(V, Constant(0.0), 'on_boundary') # homog. bcs for Newton step
        sigma_lin = lambda u, u_k: q(u_k)*grad(u) \
                                    + (u - u_k)*dqdu(u_k)*grad(u_k) \
                                    + grad(u - u_k)*inner(dqdg(u_k), grad(u_k))
        #F = (inner(sigma_lin(u, u_k), grad(v)) - f_h*v)*dx
        F = (inner(sigma(u), grad(v)) - f_h*v)*dx
        du = TrialFunction(V)
        J = derivative(F, u, du)
        du = Function(V)
    elif solver == 'Picard':
        u = TrialFunction(V)
        sigma_lin = lambda u, u_k: q(u_k)*grad(u)
        F = (inner(sigma_lin(u, u_k), grad(v)) - f_h*v)*dx
        a = lhs(F)
        L = rhs(F)
    else: print('Solver not recognised'); exit
    
    # Construct f-vector
    f_hvec = get_fvec(f_h, mesh)
    
    eta_lin = 1.
    eta_disc = 0.
    gamma_lin = 0.0000001
    maxIter = 25; dispOutput = True
    iterDiffArray = [0]
    exactErrArray = [0]
    iter = 0
    error_estimators = np.array([[0.,0.,0.,0.,0.]])
    JupArray = [0]
    # Begin Picard iterations
    while eta_lin > gamma_lin*eta_disc and iter < maxIter:
        iter += 1
        if solver == 'Newton':
            A, b = assemble_system(J, -F, bcs_du)
            solve(A, du.vector(), b)
            u.vector()[:] += du.vector()
        elif solver == 'Picard':
            solve(a == L, u, bcs)
        # calculate iterate difference and exact error in L2 norm
        itErr = errornorm(u_k, u, 'H1')
        exErr = errornorm(exact_solution, u, 'H1')
        iterDiffArray.append(itErr) # fill arrays with error data
        exactErrArray.append(exErr)
        
        eta_disc, eta_lin, eta_quad, eta_osc, eta_NC = get_estimators(V, f, f_h, f_hvec, sigma, sigma_lin, u, u_k, mesh, p)
        error_estimators = np.concatenate((error_estimators, np.array([[eta_disc, eta_lin, eta_quad, eta_osc, eta_NC]])), axis = 0)
        JupArray.append(assemble((inner(sigma(u_e) - sigma(u), sigma(u_e) - sigma(u)))**(qu/2)*dx)**(1/qu) + error_estimators[-1,4])
        if dispOutput:
            print('k = ' + str(iter) + ' | u-diff =  ' + str(itErr) + 
            ', exact error = ' + str(exErr) + ', eta_lin = ' + str(eta_lin)
            + ' , eta_disc = ' + str(eta_disc))
        u_k.assign(u) # update for next iteration

    return u, error_estimators