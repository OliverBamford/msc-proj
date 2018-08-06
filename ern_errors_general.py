from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

def left_boundary(x, on_boundary):
    return on_boundary and abs(x[0]) < 1E-14
def right_boundary(x, on_boundary):
    return on_boundary and abs(x[0]-1) < 1E-14
def top_boundary(x, on_boundary):
    return on_boundary and abs(x[1]-1) < 1E-14
def bottom_boundary(x, on_boundary):
    return on_boundary and abs(x[1]) < 1E-14
def on_boundary(x, on_boundary):
    return on_boundary and abs(x) < 1E-14
def q(u):
    #return (1+u)**2 
    return (inner(grad(u),grad(u)))**4
def dqdu(u):
    #return 2*(1+u)
    return None
def dqdgu(u):
    #return None
    return 8*inner(grad(u),grad(u))**3*grad(u)
def sigma(u):
    return q(u)*grad(u)
def sigma_Picard(u, u_k):
    return q(u_k)*grad(u)
def sigma_Newton(u, u_k):
    SN = q(u_k)*grad(u)
    if dqdu(u) != None: SN += (u - u_k)*dqdu(u_k)*grad(u_k)
    if dqdgu(u) != None: SN += grad(u - u_k)*inner(dqdgu(u_k), grad(u_k))
    return SN
            

### MESH SETUP ###
N = 15 # mesh fine-ness
l = 1 # order of finite elements
d = 2 # dimension of the space
mesh = UnitSquareMesh(N,N)
n = FacetNormal(mesh)
mesh.init()
V = FunctionSpace(mesh, 'CR', l)

### BOUNDARY CONDITIONS ###
x, y= sym.symbols('x[0] x[1]')
uExSym = 9./10.*((0.5)**(10./9.) - ((x - 0.5)**2 + (y - 0.5)**2)**(5./9.))
uExpr = Expression(sym.printing.ccode(uExSym), degree=4)
u_e = interpolate(uExpr, V)
bcs = DirichletBC(V, uExpr, 'on_boundary')

# set up BCs on left and right
# lambda functions ensure the boundary methods take two variables
#B1 = DirichletBC(V, Constant(0.0), lambda x, on_boundary : left_boundary(x, on_boundary))
#B2 = DirichletBC(V, Constant(1.0), lambda x, on_boundary : right_boundary(x, on_boundary))
#bcs = [B1, B2]
#uExpr = Expression('pow((pow(2,m+1) - 1)*x[0] + 1, (1/(m+1))) - 1', m = 2, degree=4)
#u_e = interpolate(uExpr, V)
### INITIAL GUESS ###
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(2.0)
DG0 = FunctionSpace(mesh, 'DG', 0)
f_h = interpolate(f, DG0)
#u_k = TrialFunction(V)
#a0 = inner(grad(u_k), grad(v))*dx
#L0 = f_h*v*dx
#u_k = Function(V)
#solve(a0 == L0, u_k, bcs)
#u_k = interpolate(Constant(1.0), V) # previous (known) u
u0Sym = uExSym*(1 + (x-0.3)*(y-0.3))
u_k = interpolate(Expression(sym.printing.ccode(uExSym), degree=4), V)

### PROBLEM SET UP ###
p = 10
qu = p/float(p-1)
sigma_lin = lambda u, u_k: sigma_Newton(u, u_k)
F = (inner(sigma_lin(u, u_k), grad(v)) - f_h*v)*dx
a = lhs(F)
L = rhs(F)

### Calculation of f_h vector for Ern errors ### 
# General Idea: Interpolate cartesian coords   #
# into DG space, which has pointwise evaluation#
# dofs, then subtract cell midpoints from      #
# corresponding edges. Then, interpolate       #
# into RTN.                                    #
################################################

F = VectorFunctionSpace(mesh, 'DG', 2, dim=d)
X = MeshCoordinates(mesh)
f_hvec_CR = interpolate(X, F)
f_ = np.zeros(f_hvec_CR.vector().get_local().shape) # create np array which contains values to be assigned to f_hvec

dm = F.dofmap()
for cell in cells(mesh):
    dofs = dm.cell_dofs(cell.index())
    f_c = f_hvec_CR.vector().get_local()[dofs] # get np array of all dof values in cell
    f_c *= f_h.vector().get_local()[cell.index()]/d
    mp = cell.midpoint().array() # get midpoint of cell
    f_c[0:f_c.size/2] -= f_h.vector().get_local()[cell.index()]*mp[0]/d # construct f_h in cell
    f_c[f_c.size/2:f_c.size] -= f_h.vector().get_local()[cell.index()]*mp[1]/d # note [x,x,x,y,y,y] structure of f_c
    f_[dofs] = f_c # place cell values back into main array
f_hvec_CR.vector().set_local(f_)

# we have constructed f_h in DG, need f_h in RTN
RTN = FunctionSpace(mesh, 'RT', 1) #VectorFunctionSpace(mesh, 'DG', 1, dim=d)
f_hvec = interpolate(f_hvec_CR, RTN)

F0 = VectorFunctionSpace(mesh, 'DG', degree=0, dim=d) # space for 0-order interpolants (sigma)
dm0 = F0.dofmap()
u = Function(V)
dflux_vec = Function(F)
lflux_vec = Function(F)
mf = MeshFunctionSizet(mesh, d-1, 0) # used to mark edges for integration
x_ = interpolate(X, F) # used as 'x' vector when constructing flux
               
itErr = 1.0 # error measure ||u-u_k||
eta_lin = 1.
eta_disc = 0.
gamma_lin = 0.0000001
maxIter = 25; dispOutput = True
iterDiffArray = []
exactErrArray = []
iter = 0
error_estimators = np.array([[0.,0.,0.,0.,0.]])
JupArray = []
# Begin Picard iterations
while eta_lin > gamma_lin*eta_disc and iter < maxIter:
    iter += 1
    
    solve(a == L, u, bcs, solver_parameters={"linear_solver": "lu"})
    
    # calculate iterate difference and exact error in L2 norm
    itErr = errornorm(u_k, u, 'H1')
    exErr = errornorm(uExpr, u, 'H1')
    
    iterDiffArray.append(itErr) # fill arrays with error data
    exactErrArray.append(exErr)
    
    # construct interpolation of sigma into F0
    # sigmakBar = Pi_0 sigma^{k-1}
    # sigmaBar = Pi_0 sigma
    sigmakBar0 = project(sigma_lin(u, u_k), F0)
    #sigmakBar = interpolate(sigmakBar0, RTN)
    sigmaBar0 = project(sigma(u), F0)
    #sigmaBar = interpolate(sigmaBar0, RTN)
    
    # construct sum (second terms Eqns (6.7) and (6.9) for each cell K
    # find residual for each edge using 'test function trick'
    R_eps = assemble(f_h*v*dx - inner(sigma_lin(u, u_k), grad(v))*dx)
    Rbar_eps = assemble(f_h*v*dx - inner(sigma(u), grad(v))*dx)
    rk = Function(F)
    r = Function(F)
    rk_ = np.zeros(rk.vector().get_local().shape)
    r_ = np.zeros(r.vector().get_local().shape)
    eta_disc = 0.
    eta_quad = 0.
    eta_osc = 0.
    eta_NC = 0.
    d_ = np.zeros(dflux_vec.vector().get_local().shape)
    l_ = np.zeros(lflux_vec.vector().get_local().shape)
    for cell in cells(mesh):
        dofs = dm.cell_dofs(cell.index()) # get indices of dofs belonging to cell
        dofs0 = dm0.cell_dofs(cell.index()) # and for the 0th order projections
        rk_c = rk.vector().get_local()[dofs]
        r_c = r.vector().get_local()[dofs]
        dflux_c = dflux_vec.vector().get_local()[dofs]
        lflux_c = lflux_vec.vector().get_local()[dofs]
        x_c = x_.vector().get_local()[dofs]
        myEdges = edges(cell)
        myVerts = vertices(cell)
        eps_K = [myEdges.next(), myEdges.next(), myEdges.next()]
        a_K = [myVerts.next(), myVerts.next(), myVerts.next()]
        mp = cell.midpoint().array()
        eta_NCK = 0.
        # a_K[n] is the vertex opposite to the edge eps_K[n]
        for i in range(0, len(eps_K)-1):
           cardT_e = eps_K[i].entities(d).size # number of cells sharing e
           R_e = R_eps[eps_K[i].index()][0] # find residual corresponding to edge
           Rbar_e = Rbar_eps[eps_K[i].index()][0] # find barred residual corresponding to edge          
           # find distance between all dofs on cell and vertex opposite to edge
           if eps_K[i].entities(d).size > 1: # if edge is internal
               rk_c[0:rk_c.size/2] += 1./(cardT_e*d)*R_e*(x_c[0:rk_c.size/2] - a_K[i].point().array()[0])
               rk_c[rk_c.size/2:rk_c.size] += 1./(cardT_e*d)*R_e*(x_c[rk_c.size/2:rk_c.size] - a_K[i].point().array()[1])
               r_c[0:r_c.size/2] += 1./(cardT_e*d)*Rbar_e*(x_c[0:r_c.size/2] - a_K[i].point().array()[0])
               r_c[r_c.size/2:r_c.size] += 1./(cardT_e*d)*Rbar_e*(x_c[r_c.size/2:r_c.size] - a_K[i].point().array()[1])
               # s = q
               mf.set_value(eps_K[i].mesh_id(), 1) # mark domain to integrate over
               eta_NCK += assemble((jump(u,n)**2)**(qu/2)*dS(subdomain_data=mf)) / eps_K[i].length() # Lq-norm of jump along edge^q
               mf.set_value(eps_K[i].mesh_id(), 0) # un-mark domain
        
        dflux_c = -sigmaBar0.vector().get_local()[dofs0].repeat(6) + f_hvec_CR.vector().get_local()[dofs] - r_c
        lflux_c = -sigmakBar0.vector().get_local()[dofs0].repeat(6) + f_hvec_CR.vector().get_local()[dofs] - rk_c - dflux_c
        d_[dofs] = dflux_c
        l_[dofs] = lflux_c
        eta_NCK = eta_NCK**(1/qu)
        dflux_vec.vector().set_local(d_)
        dflux = interpolate(dflux_vec, RTN) # we must do this at every iteration to calculate eta_disc
        # add local discretisation estimator^q to total
        eta_disc += (2**(1./p)*(assemble_local(((dflux+sigmaBar0)**2)**(qu/2)*dx, cell)**(1/qu) + eta_NCK))**qu
        eta_osc += cell.h()/np.pi * assemble_local(((f - f_h)**2)**(qu/2)*dx, cell)**(qu)
        eta_NC += eta_NCK
        rk_[dofs] = rk_c # place cell values back into main array
        r_[dofs] = r_c # place cell values back into main array
        
    lflux_vec.vector().set_local(l_)
    rk.vector().set_local(rk_)
    r.vector().set_local(r_)
    # interpolate CR construction of residuals into RTN
    rk = interpolate(rk, RTN)
    r = interpolate(r, RTN)
    # compute global discretisation and quadrature estimators
    eta_disc = eta_disc**(1/qu)
    eta_NC = eta_NC**(1/qu)
    eta_quad = assemble(((sigma(u)-sigmaBar0)**2)**(qu/2)*dx)**(1/qu)
    eta_osc = eta_osc**(1/qu)
    # construct flux (d+l) for each element (Eq. (6.7))
    #dflux.assign(-sigmaBar + f_hvec - r) 
    #lflux.assign(-sigmakBar + f_hvec - rk - dflux)
    lflux = interpolate(lflux_vec, RTN)
    eta_lin = assemble((lflux**2)**(qu/2)*dx)**(1/(2*qu))
    #eta_lin = norm(lflux, 'L2')**(0.5)
    JupArray.append(assemble(((sigma(u_e) - sigma(u))**2)**(qu/2)*dx)**(1/qu) + eta_NC)
    error_estimators = np.concatenate((error_estimators, np.array([[eta_disc, eta_lin, eta_quad, eta_osc, eta_NC]])), axis = 0)

    if dispOutput:
        print('k = ' + str(iter) + ' | u-diff =  ' + str(itErr) + 
        ', exact error = ' + str(exErr) + ', eta_lin = ' + str(eta_lin)
        + ' , eta_disc = ' + str(eta_disc))
    u_k.assign(u) # update for next iteration

plt.figure()
plt.semilogy(error_estimators[:,0], label='$\eta_{disc}$')
plt.semilogy(error_estimators[:,1], label='$\eta_{lin}$')
#plt.semilogy(error_estimators[:,2], label='$\eta_{quad}$')
plt.semilogy(error_estimators[:,3], label='$\eta_{osc}$')
plt.semilogy(error_estimators[:,4], label='$\eta_{NC}$')
plt.semilogy(error_estimators[:,0:4].sum(axis=1), label='$\eta$')
plt.semilogy(exactErrArray, label='$||u_h - u_e||_{H1}$')
plt.semilogy(JupArray, label='$J^{up}$')
plt.xlabel('Iterations')
plt.ylabel('$\eta$')
plt.legend()