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
        
N = 10
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

V = V
u = TrialFunction(V)
v = TestFunction(V)
u_k = interpolate(Constant(0.0), V) # previous (known) u
a = inner(q(u_k)*grad(u), grad(v))*dx
f = Constant(1.0)
L = f*v*dx

bcs = [B1, B2] 

#X = SpatialCoordinate(mesh)
DG0 = FunctionSpace(mesh, 'DG', 0)
DG1 = FunctionSpace(mesh, 'DG', 1)
f_h = interpolate(f, DG0)

myCells = cells(mesh)
centers = []
cellList = []
# create list of cell centers (dtype: Point)
for k in range(0, 2*N**d):
    cellList.append(myCells.next())
    centers.append(cellList[k].midpoint())

F = VectorFunctionSpace(mesh, 'DG', degree=0, dim=d)
# construct f_h(x)|_K vector
f_hvec = []
for k in range(0, len(cellList)):
    f_hvec.append(interpolate(Expression(['c*(x[0] - xk0)', 'c*(x[1] - xk1)'],
                                         xk0=centers[k].array()[0],
                                         xk1=centers[k].array()[1], 
                                         c = f_h(centers[k])/d , degree=1), F))
                                     
# construct residual sum
#k = 0
#for K in cellList:
#    centerK = centers[k]
#    k +=1
                                        
u = Function(V)
lflux = Function(F)
dflux = Function(F)
                         
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
                                            u = u_k, gu = gu, degree=0), F)
                                            # u = u_k (previous u iterate)
    sigmaBar = interpolate(Expression(['((1 + u)*(1 + u) + \
                                            (1 + u)*(1 + u))*gu[0]', 
                                        '((1 + u)*(1 + u) + \
                                            (1 + u)*(1 + u))*gu[1]'], 
                                            u = u, gu = gu, degree=0), F)
    # construct sum (second terms Eqns (6.7) and (6.9) for each cell K
    rSum = []
    rBarSum = []
    # find residual for each edge using 'test function trick'
    R_eps = assemble(f_h*v*dx - inner(sigmakBar, grad(v))*dx)
    Rbar_eps = assemble(f_h*v*dx - inner(sigmaBar, grad(v))*dx)
    for K in cellList:
        myEdges = edges(K)
        myVerts = vertices(K)
        # create list of cells edges and vertices 
        # (would probably be replaced by a loop in 3D case)
        eps_K = [myEdges.next(), myEdges.next(), myEdges.next()]
        a_K = [myVerts.next(), myVerts.next(), myVerts.next()]
        # |T_e| = 1 for all internal edges
        cardT_e = 1
        # a_K[n] is the vertex opposite to the edge eps_K[n]
        sumK = Function(F)
        sumBarK = Function(F)
        for i in range(0, len(eps_K)-1):
            #TODO: implement check to see if edge is on boundary
            R_e = R_eps[eps_K[i].index()][0] # find residual corresponding to edge
            Rbar_e = Rbar_eps[eps_K[i].index()][0] # find barred residual corresponding to edge
            sumK += interpolate(Expression(['C*(x[0] - a0)',
                                            'C*(x[1] - a1)'],
                                            C = R_e/(cardT_e*d), 
                                            a0 = a_K[i].point().array()[0],
                                            a1 = a_K[i].point().array()[1],
                                            degree=1), F)
            sumBarK += interpolate(Expression(['C*(x[0] - a0)',
                                            'C*(x[1] - a1)'],
                                            C = Rbar_e/(cardT_e*d), 
                                            a0 = a_K[i].point().array()[0],
                                            a1 = a_K[i].point().array()[1],
                                            degree=1), F)
        rSum.append(sumK)
        rBarSum.append(sumBarK)
        
    # construct flux (d+l) for each element (Eq. (6.7))
    d_T = []
    l_T = []
    eta_lin = 0.
    for k in range(0, len(f_hvec)):
        dflux.assign(-sigmaBar + f_hvec[k] - rBarSum[k])
        lflux.assign(-sigmakBar + f_hvec[k] - rSum[k] - dflux)
        d_T.append(dflux.copy(deepcopy=True))
        l_T.append(lflux.copy(deepcopy=True))
        eta_lin += norm(lflux, 'L2')**2
    eta_lin = eta_lin**(0.5)
    eta_linArray.append(eta_lin)
    if dispOutput:
        print('k = ' + str(iter) + ' | u-diff =  ' + str(itErr) + 
        ', exact error = ' + str(exErr) + ', eta_lin = ' + str(eta_lin))
        
    u_k.assign(u) # update for next iteration