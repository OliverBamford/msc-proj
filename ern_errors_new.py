from fenics import *
import matplotlib.pyplot as plt
import numpy as np

def left_boundary(x, on_boundary):
    return on_boundary and abs(x[0]) < 1E-14
def right_boundary(x, on_boundary):
    return on_boundary and abs(x[0]-1) < 1E-14
def on_boundary(x, on_boundary):
    return on_boundary and abs(x) < 1E-14
def q(u):
    return (1+u)**2 
    #return (inner(grad(u),grad(u)))**4
def dqdu(u):
    return 2*(1+u)
    #return None
def dqdgu(u):
    return None
    #return 8*inner(grad(u),grad(u))**3*grad(u)
def sigma(u):
    return q(u)*grad(u)
def sigma_Picard(u, u_k):
    return q(u_k)*grad(u)
def sigma_Newton(u, u_k):
    SN = q(u_k)*grad(u)
    if dqdu(u) != None: SN += (u - u_k)*dqdu(u_k)*grad(u_k)
    if dqdgu(u) != None: SN += grad(u - u_k)*inner(dqdgu(u_k), grad(u_k))
    return SN

N = 5
l = 1
d = 2 # dimension of the space

mesh = UnitSquareMesh(N,N)
V = FunctionSpace(mesh, 'CR', l)

# set up BCs on left and right
# lambda functions ensure the boundary methods take two variables
B1 = DirichletBC(V, Constant(0.0), lambda x, on_boundary : left_boundary(x, on_boundary)) # u(0) = 0
B2 = DirichletBC(V, Constant(1.0), lambda x, on_boundary : right_boundary(x, on_boundary)) # u(1) = 1

# construct exact solution in C++ format
uExpr = Expression('pow((pow(2,m+1) - 1)*x[0] + 1, (1/(m+1))) - 1', m = 2, degree=4)

iterTol = 1.0e-11; maxIter = 25; dispOutput = True

u = TrialFunction(V)
v = TestFunction(V)
u_k = interpolate(Constant(1.0), V) # previous (known) u
f = Constant(0.0)


sigma_lin = lambda u, u_k: sigma_Picard(u, u_k)

bcs = [B1, B2] 

#X = SpatialCoordinate(mesh)
DG0 = FunctionSpace(mesh, 'DG', 0)
f_h = interpolate(f, DG0)

F = (inner(sigma_lin(u, u_k), grad(v)) - f_h*v)*dx
a = lhs(F)
L = rhs(F)

F = VectorFunctionSpace(mesh, 'CR', 1, dim=d)
X = MeshCoordinates(mesh)
f_hvec_CR = interpolate(X, F)
f_ = np.zeros(f_hvec_CR.vector().get_local().shape) # create np array which contains values to be assigned to f_hvec

dm = F.dofmap()
for cell in cells(mesh):
    dofs = dm.cell_dofs(cell.index())
    f_c = f_hvec_CR.vector().get_local()[dofs] # get np array of all dof values in cell
    mp = cell.midpoint().array() # get midpoint of cell
    f_c[0:f_c.size/2] = f_h.vector().get_local()[cell.index()]*(f_c[0:f_c.size/2] - mp[0])/d # construct f_h in cell
    f_c[f_c.size/2:f_c.size] = f_h.vector().get_local()[cell.index()]*(f_c[f_c.size/2:f_c.size] - mp[1])/d # note [x,x,x,y,y,y] structure of f_c
    f_[dofs] = f_c # place cell values back into main array

f_hvec_CR.vector().set_local(f_)

# we have constructed f_h in CR, which has dofs: pointwise evaluation
# need to constuct f_h in RTN, which has dofs: moments of the normal component against P_{q-1} on facets
# note that dofs are in the same locations for CR1 and RT1
RTN = FunctionSpace(mesh, 'RT', 1)
f_hvec = interpolate(f_hvec_CR, RTN)

F0 = VectorFunctionSpace(mesh, 'DG', degree=0, dim=d) # space for 0-order interpolants (sigma)
dm0 = F0.dofmap()
u = Function(V)
lflux = Function(RTN)
dflux = Function(RTN)
dflux_vec = Function(F)
lflux_vec = Function(F)
discflux = Function(RTN)
mf = MeshFunctionSizet(mesh, 1, 0)
x_ = interpolate(X, F) # used as 'x' vector when constructing flux
               
itErr = 1.0 # error measure ||u-u_k||
eta_lin = 1.
eta_disc = 0.
gamma_lin = 0.0000001
iterDiffArray = []
exactErrArray = []
iter = 0
error_estimators = np.array([[0.,0.,0.,0.]])
# Begin Picard iterations
while eta_lin > gamma_lin*eta_disc and iter < maxIter:
    iter += 1
    
    solve(a == L, u, bcs, solver_parameters={"linear_solver": "lu"})
    
    # calculate iterate difference and exact error in L2 norm
    itErr = errornorm(u_k, u, 'H1')
    exErr = errornorm(uExpr, u, 'H1')
    
    iterDiffArray.append(itErr) # fill arrays with error data
    exactErrArray.append(exErr)
    
    sigmakBar0 = project(sigma_lin(u, u_k), F0)
    sigmakBar = interpolate(sigmakBar0, RTN)
    sigmaBar0 = project(sigma(u), F0)
    sigmaBar = interpolate(sigmaBar0, RTN)
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
    d_ = np.zeros(dflux_vec.vector().get_local().shape)
    l_ = np.zeros(lflux_vec.vector().get_local().shape)
    for cell in cells(mesh):
        dofs = dm.cell_dofs(cell.index()) # get indices of dofs belonging to cell
        dofs0 = dm0.cell_dofs(cell.index()) # and for the 0th order projections
        rk_c = rk.vector().get_local()[dofs]
        r_c = r.vector().get_local()[dofs]
        x_c = x_.vector().get_local()[dofs]
        dflux_c = dflux_vec.vector().get_local()[dofs]
        lflux_c = lflux_vec.vector().get_local()[dofs]
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
           rk_c[0:rk_c.size/2] += 1./(cardT_e*d)*R_e*(x_c[0:rk_c.size/2] - a_K[i].point().array()[0]) 
           rk_c[rk_c.size/2:rk_c.size] += 1./(cardT_e*d)*R_e*(x_c[rk_c.size/2:rk_c.size] - a_K[i].point().array()[1]) 
           r_c[0:r_c.size/2] += 1./(cardT_e*d)*Rbar_e*(x_c[0:r_c.size/2] - a_K[i].point().array()[0]) 
           r_c[r_c.size/2:r_c.size] += 1./(cardT_e*d)*Rbar_e*(x_c[r_c.size/2:r_c.size] - a_K[i].point().array()[1]) 
           
           if eps_K[i].entities(d).size > 1: # if edge is internal
               # s := q = 2
               mf.set_value(eps_K[i].mesh_id(), 1) # mark domain to integrate over
               eta_NCK += assemble(jump(u)*jump(u)*dS(subdomain_data=mf)) / eps_K[i].length() # squared L2 norm of jump along edge
               mf.set_value(eps_K[i].mesh_id(), 0) # un-mark domain
               
        dflux_c = -sigmaBar0.vector().get_local()[dofs0].repeat(3) + f_hvec_CR.vector().get_local()[dofs] - r_c
        lflux_c = -sigmakBar0.vector().get_local()[dofs0].repeat(3) + f_hvec_CR.vector().get_local()[dofs] - rk_c - dflux_c
        d_[dofs] = dflux_c
        l_[dofs] = lflux_c
        dflux_vec.vector().set_local(d_)
        dflux = interpolate(dflux_vec, RTN) # we must do this at every iteration to calculate eta_disc
        # add squared local discretisation estimator to total
        eta_disc += (2**(0.5)*(assemble_local((dflux+sigmaBar)**2*dx, cell)**(0.5) + eta_NCK))**2
        eta_osc += cell.h()/np.pi * assemble_local((f - f_h)**2*dx, cell)**(0.5)
        rk_[dofs] = rk_c # place cell values back into main array
        r_[dofs] = r_c # place cell values back into main array
        
    lflux_vec.vector().set_local(l_)    
    rk.vector().set_local(rk_)
    r.vector().set_local(r_)
    
    # interpolate CR construction of residuals into RTN
    rk = interpolate(rk, RTN)
    r = interpolate(r, RTN)    
    # compute global discretisation and quadrature estimators
    eta_disc = eta_disc**(0.5)
    eta_quad = assemble((sigma(u)-sigmaBar)**2*dx)**(0.5)
    # construct flux (d+l) for each element (Eq. (6.7))
#    dflux.assign(-sigmaBar + f_hvec - r)
#    lflux.assign(-sigmakBar + f_hvec - rk - dflux)
    
    lflux = interpolate(lflux_vec, RTN)
    eta_lin = assemble((lflux**2)**(qu/2)*dx)**(1/(2*qu))
   
    error_estimators = np.concatenate((error_estimators, np.array([[eta_disc, eta_lin, eta_quad, eta_osc]])), axis = 0)

    if dispOutput:
        print('k = ' + str(iter) + ' | u-diff =  ' + str(itErr) + 
        ', exact error = ' + str(exErr) + ', eta_lin = ' + str(eta_lin)
        + ' , eta_disc = ' + str(eta_disc))
    u_k.assign(u) # update for next iteration

plt.figure()
plt.semilogy(error_estimators[:,0], label='$\eta_{disc}$')
plt.semilogy(error_estimators[:,1], label='$\eta_{lin}$')
plt.semilogy(error_estimators[:,2], label='$\eta_{quad}$')
plt.semilogy(error_estimators[:,3], label='$\eta_{osc}$')
plt.semilogy(error_estimators.sum(axis=1), label='$\eta$')
plt.semilogy(exactErrArray, label='$J_{u}$')
plt.xlabel('Iterations')
plt.ylabel('$\eta$')
plt.legend()