from fenics import *
import matplotlib.pyplot as plt
import numpy as np

def left_boundary(x, on_boundary):
        return on_boundary and abs(x[0]) < 1E-14
def right_boundary(x, on_boundary):
        return on_boundary and abs(x[0]-1) < 1E-14
def q(u):
        return (1+u)**2
def dqdu(u):
        return 2*(1+u)
        
N = 5
p = 1
d = 2 # dimension of the space

mesh = UnitSquareMesh(N,N)
mesh.init(1) # generate edges
V = FunctionSpace(mesh, 'CR', p)

# set up BCs on left and right
# lambda functions ensure the boundary methods take two variables
B1 = DirichletBC(V, Constant(0.0), lambda x, on_boundary : left_boundary(x, on_boundary)) # u(0) = 0
B2 = DirichletBC(V, Constant(1.0), lambda x, on_boundary : right_boundary(x, on_boundary)) # u(1) = 1

# construct exact solution in C++ format
uExpr = Expression('pow((pow(2,m+1) - 1)*x[0] + 1,(1/(m+1))) - 1', m = 2, degree=4)

iterTol = 1.0e-5; maxIter = 25; dispOutput = True

u = TrialFunction(V)
v = TestFunction(V)
u_k = interpolate(Constant(0.0), V) # previous (known) u
a = inner(q(u_k)*grad(u), grad(v))*dx
f = Constant(1.0)
L = f*v*dx

bcs = [B1, B2] 

#X = SpatialCoordinate(mesh)
DG0 = FunctionSpace(mesh, 'DG', 0)
f_h = interpolate(f, DG0)
#TODO: construct fluxes in CR space NOT DG space (f_h should be in DG0 still)
F = VectorFunctionSpace(mesh, 'CR', degree=1, dim=d)
f_hvec = interpolate(Expression(['x[0]', 'x[1]'], degree=1), F)
f_ = np.zeros(f_hvec.vector().get_local().shape) # create np array which contains values to be assigned to f_hvec

dm = F.dofmap()
for cell in cells(mesh):
    dofs = dm.cell_dofs(cell.index())
    f_c = f_hvec.vector().get_local()[dofs] # get np array of all dofs in cell
    mp = cell.midpoint().array() # get midpoint of cell
    f_c[0:f_c.size/2] = f*(f_c[0:f_c.size/2] - mp[0])/d # construct f_h in cell
    f_c[f_c.size/2:f_c.size] = f*(f_c[f_c.size/2:f_c.size] - mp[1])/d # note [x,x,x,y,y,y] structure of f_c
    f_[dofs] = f_c # place cell values back into main array       

f_hvec.vector().set_local(f_)

F0 = VectorFunctionSpace(mesh, 'DG', degree=0, dim=d) # space for 0-order interpolants (sigma)             
u = Function(V)
lflux = Function(F)
dflux = Function(F)
x_ = interpolate(Expression(['x[0]', 'x[1]'], degree=0), F) # used as 'x' vector when constructing flux
                
itErr = 1.0           # error measure ||u-u_k||
iterDiffArray = []
exactErrArray = []
eta_linArray = []
iter = 0

# Begin Picard iterations
while itErr > iterTol and iter < maxIter:
    iter += 1
    
    solve(a == L, u, bcs, solver_parameters={"linear_solver": "lu"})
    
    # calculate iterate difference and exact error in L2 norm
    itErr = errornorm(u_k, u, 'H1')
    exErr = errornorm(uExpr, u, 'H1')
    
    iterDiffArray.append(itErr) # fill arrays with error data
    exactErrArray.append(exErr)
    
    # construct interpolation of sigma = (1+u)^2 * grad(u) into F 
    # sigma^{k-1} = (1+u_k)**2 * grad(u)
    # sigma = (1+u)**2 * grad(u)
    # sigmakBar = Pi_0 sigma^{k-1}
    # sigmaBar = Pi_0 sigma
    gu = project(grad(u), F)
    sigmakBar = interpolate(Expression(['((1 + u)*(1 + u) + \
                                            (1 + u)*(1 + u))*gu[0]', 
                                        '((1 + u)*(1 + u) + \
                                            (1 + u)*(1 + u))*gu[1]'], 
                                            u = u_k, gu = gu, degree=0), F0)
                                            # u = u_k (previous u iterate)
    # must be interpolated into higher order CR space to be used for flux calculation
    sigmakBar = interpolate(sigmakBar, F) 
    sigmaBar = interpolate(Expression(['((1 + u)*(1 + u) + \
                                            (1 + u)*(1 + u))*gu[0]', 
                                        '((1 + u)*(1 + u) + \
                                            (1 + u)*(1 + u))*gu[1]'], 
                                            u = u, gu = gu, degree=0), F0)
    sigmaBar = interpolate(sigmaBar, F)
    # construct sum (second terms Eqns (6.7) and (6.9) for each cell K
    # find residual for each edge using 'test function trick'
    R_eps = assemble(f_h*v*dx - inner(sigmakBar, grad(v))*dx)
    Rbar_eps = assemble(f_h*v*dx - inner(sigmaBar, grad(v))*dx)
    rk = Function(F)
    r = Function(F)
    rk_ = np.zeros(rk.vector().get_local().shape)
    r_ = np.zeros(r.vector().get_local().shape)
    jumps = 0
    for cell in cells(mesh):
        dofs = dm.cell_dofs(cell.index())
        rk_c = rk.vector().get_local()[dofs]
        r_c = r.vector().get_local()[dofs]
        x_c = x_.vector().get_local()[dofs]
        myEdges = edges(cell)
        myVerts = vertices(cell)
        eps_K = [myEdges.next(), myEdges.next(), myEdges.next()]
        a_K = [myVerts.next(), myVerts.next(), myVerts.next()]
        # |T_e| = 1 for all internal edges
        cardT_e = 1
        # a_K[n] is the vertex opposite to the edge eps_K[n]
        for i in range(0, len(eps_K)-1):
           R_e = R_eps[eps_K[i].index()][0] # find residual corresponding to edge
           Rbar_e = Rbar_eps[eps_K[i].index()][0] # find barred residual corresponding to edge          
           # find distance between all dofs on cell and vertex opposite to edge
           rk_c[0:rk_c.size/2] += 1./(cardT_e*d)*R_e*(x_c[0:rk_c.size/2] - a_K[i].point().array()[0]) 
           rk_c[rk_c.size/2:rk_c.size] += 1./(cardT_e*d)*R_e*(x_c[rk_c.size/2:rk_c.size] - a_K[i].point().array()[1]) 
           r_c[0:r_c.size/2] += 1./(cardT_e*d)*Rbar_e*(x_c[0:r_c.size/2] - a_K[i].point().array()[0]) 
           r_c[r_c.size/2:r_c.size] += 1./(cardT_e*d)*Rbar_e*(x_c[r_c.size/2:r_c.size] - a_K[i].point().array()[1]) 
           
        rk_[dofs] = rk_c # place cell values back into main array
        r_[dofs] = r_c # place cell values back into main array
    rk.vector().set_local(rk_)
    r.vector().set_local(r_)
    # construct flux (d+l) for each element (Eq. (6.7))
    dflux.assign(-sigmaBar + f_hvec - r)
    lflux.assign(-sigmakBar + f_hvec - rk - dflux)
    eta_lin = norm(lflux, 'L2')**(0.5)
    eta_linArray.append(eta_lin)
    
    if dispOutput:
        print('k = ' + str(iter) + ' | u-diff =  ' + str(itErr) + 
        ', exact error = ' + str(exErr) + ', eta_lin = ' + str(eta_lin))
    u_k.assign(u) # update for next iteration