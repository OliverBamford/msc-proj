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
        mp = cell.midpoint().array()
        eta_NCK = 0.
        # a_K[n] is the vertex opposite to the edge eps_K[n]
        for i in range(len(eps_K)):
           cardT_e = eps_K[i].entities(d).size # number of cells sharing e
           R_e = R_eps[eps_K[i].index()][0] # find residual corresponding to edge #TODO: check this over
           Rbar_e = Rbar_eps[eps_K[i].index()][0] # find barred residual corresponding to edge          
           # find distance between all dofs on cell and vertex opposite to edge
           #if eps_K[i].entities(d).size > 1: # if edge is internal
           rk_c[0:rk_c.size/2] += 1./(cardT_e*d)*R_e*(x_c[0:rk_c.size/2] - mp[0])
           rk_c[rk_c.size/2:rk_c.size] += 1./(cardT_e*d)*R_e*(x_c[rk_c.size/2:rk_c.size] - mp[1])
           r_c[0:r_c.size/2] += 1./(cardT_e*d)*Rbar_e*(x_c[0:r_c.size/2] - mp[0])
           r_c[r_c.size/2:r_c.size] += 1./(cardT_e*d)*Rbar_e*(x_c[r_c.size/2:r_c.size] - mp[1])
           # s = q
           mf.set_value(eps_K[i].index(), 1) # mark domain to integrate over
           eta_NCK += assemble(inner(jump(u,n), jump(u,n))**(qu/2)*dS(1, subdomain_data=mf)) / eps_K[i].length() # Lq-norm of jump along edge^q
           #TODO: make this edge integration work
           mf.set_value(eps_K[i].index(), 0) # un-mark domain
        
        dflux_c = -sigmaBar0.vector().get_local()[dofs0].repeat(3) + f_hvec.vector().get_local()[dofs] - r_c
        lflux_c = -sigmakBar0.vector().get_local()[dofs0].repeat(3) + f_hvec.vector().get_local()[dofs] - rk_c - dflux_c
        d_[dofs] = dflux_c
        l_[dofs] = lflux_c
        eta_NCK = eta_NCK**(1/qu)
        dflux.vector().set_local(d_)
        # add local discretisation estimator^q to total
        eta_disc += (2**(1./p)*(assemble_local((inner(dflux+sigmaBar0,dflux+sigmaBar0))**(qu/2)*dx, cell)**(1/qu) + eta_NCK))
        eta_disc_noNC += 2**(1./p)*assemble_local((inner(dflux+sigmaBar0,dflux+sigmaBar0))**(qu/2)*dx, cell)**(1/qu)
        eta_osc += cell.h()/np.pi*assemble_local((inner(f - f_h, f - f_h))**(qu/2)*dx, cell)
        eta_NC += eta_NCK
        rk_[dofs] = rk_c # place cell values back into main array
        r_[dofs] = r_c # place cell values back into main array
    
    lflux.vector().set_local(l_)
    rk.vector().set_local(rk_)
    r.vector().set_local(r_)
    # interpolate CR construction of residuals into RTN
    #rk = interpolate(rk, RTN)
    #r = interpolate(r, RTN)
    # compute global discretisation and quadrature estimators
    eta_disc = eta_disc**(1/qu)
    eta_disc_noNC = eta_disc_noNC**(1/qu)
    print(eta_disc_noNC)
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