from fenics import *
import numpy as np

def get_PDECO_fvecs(f_funcs, f_lin_funcs, mesh):
    """Calculation of f_h vector for Ern errors
        General Idea: Interpolate cartesian coords
        into DG space, which has pointwise evaluation
        dofs, then subtract cell midpoints from
        corresponding edges.
        Inputs:
        f_funcs: list of projections of f functions into DG0
        f_lin_funcs: list of projections of linearised f functions into DG0
    """
    f_hvecs = []
    f_lin_hvecs = []
    for k, f_h in enumerate(f_funcs):
        d = mesh.geometry().dim()
        RT = VectorFunctionSpace(mesh, 'DG', 1, dim=d)
        X = MeshCoordinates(mesh)
        f_hvec = interpolate(X, RT)
        f_lin_hvec = Function(RT)
        f_lin_hvec.assign(f_hvec)
        
        f_ = np.zeros(f_hvec.vector().get_local().shape) # create np array which contains values to be assigned to f_hvec
        f_lin_ = np.zeros(f_hvec.vector().get_local().shape)
        dm = RT.dofmap()
        for cell in cells(mesh):
            dofs = dm.cell_dofs(cell.index())
            f_c = f_hvec.vector().get_local()[dofs] # get np array of all dof values in cell
            f_lin_c = f_lin_hvec.vector().get_local()[dofs]
            
            f_c *= f_h.vector().get_local()[cell.index()]/d
            f_lin_c *= f_lin_funcs[k].vector().get_local()[cell.index()]/d
            mp = cell.midpoint().array()[0:2] # get midpoint of cell
            f_c -= f_h.vector().get_local()[cell.index()]*mp.repeat(3)/d
            f_lin_c -= f_h.vector().get_local()[cell.index()]*mp.repeat(3)/d
            
            f_[dofs] = f_c # place cell values back into main array
            f_lin_[dofs] = f_lin_c
        f_hvec.vector().set_local(f_)
        f_hvecs.append(f_hvec)
        f_lin_hvec.vector().set_local(f_lin_)
        f_lin_hvecs.append(f_lin_hvec)
    return f_hvecs, f_lin_hvecs

def get_estimators_PDECO(H, subspace_index, f, f_h, f_lin_h, f_hvec, f_lin_hvec, sigma, sigma_lin, u, u_k, U, mesh, p, bcs=None):
    V = H.sub(subspace_index)
    tf = [None, None, None]
    tf[0],tf[1],tf[2] = TestFunction(H)
    v = tf[subspace_index] 
    
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
    # unbarred residual (from Ern 2015) should be ~zero if everything is working correctly
    R_eps = assemble(f_lin_h*v*dx - inner(sigma_lin(u, u_k), grad(v))*dx)
    Rbar_eps = assemble(f_h*v*dx - inner(sigma(u), grad(v))*dx)
    if type(bcs) == list:
        for bc in bcs: # so we can handle lists of bcs
            bc.apply(R_eps, U.vector()) # apply bcs to residuals
            bc.apply(Rbar_eps, U.vector())
    elif bcs != None:
        bcs.apply(R_eps, U.vector()) # apply bcs to residuals
        bcs.apply(Rbar_eps, U.vector())

    rk = Function(RT)
    r = Function(RT)
    rk_ = np.zeros(rk.vector().get_local().shape)
    r_ = np.zeros(r.vector().get_local().shape)
    eta_disc = 0.
    eta_quad = 0.
    eta_osc = 0.
    eta_NC = 0.
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
        mp = cell.midpoint().array()[0:2] # midpoint is returned in (x,y,z) format
        eta_NCK = 0.
        # a_K[n] is the vertex opposite to the edge eps_K[n]
        for i, e_K in enumerate(edges(cell)):
           cardT_e = e_K.entities(d).size # number of cells sharing e
           R_e = R_eps[edge_dofs[e_K.index()]][0] # find residual corresponding to edge
           Rbar_e = Rbar_eps[edge_dofs[e_K.index()]][0] # find barred residual corresponding to edge          
           # find distance between all dofs on cell and cell midpoint (see Remark 6.7 of Ern)
           rk_c += 1./(cardT_e*d)*R_e*(x_c - mp.repeat(3))
           r_c += 1./(cardT_e*d)*Rbar_e*(x_c - mp.repeat(3))
           # s = q
           mf.set_value(e_K.index(), 1) # mark domain to integrate over
           eta_NCK += assemble(inner(jump(u,n), jump(u,n))**(qu/2)*dS(1, subdomain_data=mf)) / e_K.length() # Lq-norm of jump along edge^q
           mf.set_value(e_K.index(), 0) # un-mark domain
        
        dflux_c = -sigmaBar0.vector().get_local()[dofs0].repeat(3) + f_hvec.vector().get_local()[dofs] - r_c
        lflux_c = -sigmakBar0.vector().get_local()[dofs0].repeat(3) + f_lin_hvec.vector().get_local()[dofs] - rk_c - dflux_c
        d_[dofs] = dflux_c
        l_[dofs] = lflux_c
        dflux.vector().set_local(d_)
        # add local estimators^q to totals
        eta_disc += (2**(1./p)*(assemble_local((inner(dflux+sigmaBar0,dflux+sigmaBar0))**(qu/2)*dx, cell)**(1/qu) + eta_NCK**(1/qu)))**qu
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
    eta_lin = assemble(inner(lflux,lflux)**(qu/2)*dx)**(1/(2*qu))
    return eta_disc, eta_lin, eta_quad, eta_osc, eta_NC