from ern_functions_PDECO import *
from ern_functions import *
from fenics import *
import matplotlib.pyplot as plt
import numpy as np

class nonlinearPDECO:
    def __init__(self, N, ue = Expression('sin(pi*x[0])*sin(pi*x[1])', degree=3), alpha = 1e-02):
        """
        Sets up the 'hello world' PDE-constrained optimisation problem
        
        Inputs:
        N: number of finite elements in mesh
        p: order of function space
        ud: Desired distribution (UFL expression)
        alpha: regularisation parameter
        """
        # set up problem (parts common to all iterative methods)
        mesh = UnitSquareMesh(N, N)
        V = FunctionSpace(mesh, 'CR', 1)
        def left_boundary(x, on_boundary):
            return on_boundary and abs(x[0]) < 1E-14
        def right_boundary(x, on_boundary):
            return on_boundary and abs(x[0]-1) < 1E-14
        B1 = DirichletBC(V, Constant(0.0), lambda x, on_boundary : left_boundary(x, on_boundary))
        B2 = DirichletBC(V, Constant(1.0), lambda x, on_boundary : right_boundary(x, on_boundary))
        bcs = [B1, B2]
        
        exact_solution = Expression('pow((pow(2,m+1) - 1)*x[0] + 1, (1/(m+1))) - 1', m = 2, degree=4)
        
        q = lambda u: (1+u)**2
        dqdu = lambda u, u_k: 2*(1+u_k)*(u-u_k)

        # initialise problem variables
        u = TrialFunction(V)
        lmbd = TrialFunction(V)
        GJ = TrialFunction(V)
        v = TestFunction(V)
        
        self.N = N
        self.p = p
        self.ue=ue
        self.alpha = alpha
        self.mesh = mesh
        self.V = V
        self.bcs = bcs
        self.u = u
        self.lmbd = lmbd
        self.GJ = GJ
        self.v = v
    def q(self, u):
        return (1+u)**2
    def dqdu(self, u):
        return 2*(1+u)
    def djdu(self, u, ud):
        return 2*(u-ud)
#    def dqdu(u, u_k):
#        return 2*(1+u_k)*(u-u_k)
    def calculateRef(self):
        """
        Solves the PDE-constrained opt. problem using monolithic approach.
        LU factorisation is used to solve the matrix equation.

        Outputs:
        u: optimal state function
        m: optimal control function
        lmbd: Lagrange multiplier
        """
        # construct mixed function space
        mesh = self.mesh
        alpha = self.alpha
        
        P1 = FiniteElement('CR', mesh.ufl_cell(), degree=1)
        TH = MixedElement([P1, P1, P1])
        H = FunctionSpace(mesh, TH)
        
        def left_boundary(x, on_boundary):
            return on_boundary and abs(x[0]) < 1E-14
        def right_boundary(x, on_boundary):
            return on_boundary and abs(x[0]-1) < 1E-14
        B1 = DirichletBC(H.sub(0), Constant(0.0), lambda x, on_boundary : left_boundary(x, on_boundary))
        B2 = DirichletBC(H.sub(0), Constant(1.0), lambda x, on_boundary : right_boundary(x, on_boundary))
        bcs = [B1, B2]
        
        U = Function(H)
        U_k = Function(H)
        u, lmbd, f = split(U)
        u_k, lmbd_k, f_k = split(U_k)
        phi,psi,chi = TestFunction(H)
        ud = interpolate(self.ue , H.sub(0).collapse()) # interpolate over u-space
        # get initial guesses
        L0 = (u-ud)**2*dx + alpha/2*f**2*dx + inner(grad(lmbd),grad(u))*dx-f*lmbd*dx
        F0 = derivative(L0, U, TrialFunction(H))
        solve(F0==0, U, bcs)
#        L = (u-ud)**2*dx + alpha/2*f**2*dx + self.q(u)*inner(grad(lmbd),grad(u))*dx-f*lmbd*dx
#        F = derivative(L, U, TrialFunction(H))
#        solve(F == 0, U, bcs)
#        
        F = (self.q(u)*inner(grad(u),grad(phi)) - f*phi \
        + self.q(u)*inner(grad(lmbd),grad(psi)) + self.djdu(u,ud)*psi \
        + self.dqdu(u)*psi*inner(grad(lmbd),grad(u)) + (lmbd - alpha*f)*chi)*dx
        
        # load initial guesses
        assign(U_k, U)
        
        U = TrialFunction(H)
        u, lmbd, f = split(U)
        #TODO: double check this
        F_lin = (inner(self.q(u_k)*grad(u) + self.dqdu(u_k)*(u - u_k)*grad(u_k), grad(phi)) \
                + inner(self.q(u_k)*grad(lmbd) + self.dqdu(u_k)*(u - u_k)*grad(lmbd_k), grad(psi)) \
                - f*phi + 2*(u - u_k)*(1 + inner(grad(lmbd_k),grad(u_k)))*psi  \
                + self.dqdu(u_k)*inner(grad(lmbd),grad(u_k))*psi + self.djdu(u_k, ud)*psi \
                + self.dqdu(u_k)*inner(grad(lmbd_k), grad(u-u_k))*psi \
               # + self.dqdu(u_k)*inner(grad(lmbd-lmbd_k), grad(u_k))*psi \
                - alpha*f*chi + lmbd*chi)*dx
        a = lhs(F_lin); L = rhs(F_lin)
        
        U = Function(H)
        u, lmbd, f = split(U)
        
        error_estimators_state = np.array([[0, 0, 0, 0, 0]])
        error_estimators_adj = np.array([[0, 0, 0, 0, 0]])
        error_estimators_f = np.array([[0, 0, 0, 0, 0]])
        iter = 0
        while iter < 10:
            iter += 1
            solve(a == L, U, bcs)

            sigma_1 = lambda u: self.q(u)*grad(u)
            sigma_lin_1 = lambda u, u_k: self.q(u_k)*grad(u) + self.dqdu(u_k)*(u - u_k)*grad(u_k)
            f_1 = f
            f_lin_1 = f
            sigma_2 = lambda lmbd: self.q(u)*grad(lmbd)
            sigma_lin_2 = lambda lmbd, lmbd_k: self.q(u_k)*grad(lmbd) + self.dqdu(u_k)*(u - u_k)*grad(lmbd_k)
            f_2 = -self.djdu(u, ud) - self.dqdu(u)*inner(grad(lmbd),grad(u))
            f_lin_2 = -self.djdu(u_k, ud) - 2*(u - u_k)*(1 + inner(grad(lmbd_k),grad(u_k))) \
                  - self.dqdu(u_k)*inner(grad(lmbd_k),grad(u)) - self.dqdu(u_k)*inner(grad(lmbd-lmbd_k),grad(u_k))
            sigma_3 = lambda u: Constant(0.0)*grad(u) # zero function
            sigma_lin_3 = lambda u, u_k: Constant(0.0)*grad(u)
            f_3 = alpha*f - lmbd
            f_lin_3 = alpha*f - lmbd
            
            DG0 = FunctionSpace(mesh, 'DG', 0)
            f_1_h = project(f_1, DG0)
            f_lin_1_h = project(f_lin_1, DG0)
            f_2_h = project(f_2, DG0)
            f_lin_2_h = project(f_lin_2, DG0)
            f_3_h = project(f_3, DG0)
            f_lin_3_h = project(f_lin_3, DG0)
            
            f_vecs, f_lin_vecs = get_PDECO_fvecs([f_1_h,f_2_h,f_3_h], [f_lin_1_h, f_lin_2_h, f_lin_3_h], mesh)
            
            _u, _lmbd, _f = U.split(deepcopy=True) #create copies which can be used in calculations
            _u_k, _lmbd_k, _f_k = U_k.split(deepcopy=True) 
            
            eta_disc, eta_lin, eta_quad, eta_osc, eta_NC =\
            get_estimators_PDECO(H, 0, f_1, f_1_h, f_lin_1_h, f_vecs[0], f_lin_vecs[0], sigma_1, sigma_lin_1, _u, _u_k, U, mesh, 2, bcs)
            error_estimators_state = np.concatenate((error_estimators_state, np.array([[eta_disc, eta_lin, eta_quad, eta_osc, eta_NC]])), axis = 0)
            eta_disc, eta_lin, eta_quad, eta_osc, eta_NC =\
            get_estimators_PDECO(H, 1, f_2, f_2_h, f_lin_2_h, f_vecs[1], f_lin_vecs[1], sigma_2, sigma_lin_2, _lmbd, _lmbd_k, U, mesh, 2, bcs)
            error_estimators_adj = np.concatenate((error_estimators_adj, np.array([[eta_disc, eta_lin, eta_quad, eta_osc, eta_NC]])), axis = 0)
            eta_disc, eta_lin, eta_quad, eta_osc, eta_NC =\
            get_estimators_PDECO(H, 2, f_3, f_3_h, f_lin_3_h, f_vecs[2], f_lin_vecs[2], sigma_3, sigma_lin_3, _f, _f_k, U, mesh, 2, bcs)
            error_estimators_f = np.concatenate((error_estimators_f, np.array([[eta_disc, eta_lin, eta_quad, eta_osc, eta_NC]])), axis = 0)            
            print('k = ' + str(iter))
            # update iterates
            assign(U_k, U)
            
        #project solutions into finite element space (I am not sure why I have to do this...)
        self.uRef = project(u , H.sub(0).collapse())
        self.lmbdRef = project(lmbd , H.sub(1).collapse())
        self.fRef = project(f , H.sub(2).collapse())
        
        self.error_estimators_state = error_estimators_state
        self.error_estimators_adj = error_estimators_adj
        self.error_estimators_f = error_estimators_f
        
        return self.uRef, self.fRef, self.lmbdRef
myOpt = nonlinearPDECO(10)
myOpt.calculateRef()
error_estimators1 = myOpt.error_estimators_state
error_estimators2 = myOpt.error_estimators_adj
error_estimators3 = myOpt.error_estimators_f

from matplotlib import rc
import matplotlib.pylab as plt
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

plt.figure(1, figsize=(12,10))
plt.semilogy(error_estimators1[:,0],  'r^-', linewidth=2, markersize=10, label='$\eta_{disc}$')
plt.semilogy(error_estimators1[:,1],  'b^-', linewidth=2, markersize=10, label='$\eta_{lin}$')
plt.semilogy(error_estimators1[:,2], 'm^-', linewidth=2, markersize=10, label='$\eta_{quad}$')
plt.semilogy(error_estimators1[:,3], label='$\eta_{osc}$')
plt.semilogy(error_estimators1[:,4], label='$\eta_{NC}$')
plt.semilogy(error_estimators1[:,0:4].sum(axis=1),  'g^-', linewidth=2, markersize=10, label='$\eta$')
#plt.semilogy(Jup1[0:(pr[-1]+1)],  'k^-', linewidth=2, markersize=10, label='$J_u^{up}$')
#plt.semilogy(error_estimators1[:,0]*0.1,  'r-', linewidth=1.5, markersize=10, label='$\gamma_{lin}\eta_{disc}$')
plt.xlabel('Iterations', fontsize=40)
plt.ylabel('Dual error (state)', fontsize=40)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(loc=0, fontsize=30)

plt.figure(2, figsize=(12,10))
plt.semilogy(error_estimators2[:,0],  'r^-', linewidth=2, markersize=10, label='$\eta_{disc}$')
plt.semilogy(error_estimators2[:,1],  'b^-', linewidth=2, markersize=10, label='$\eta_{lin}$')
plt.semilogy(error_estimators2[:,2], 'm^-', linewidth=2, markersize=10, label='$\eta_{quad}$')
plt.semilogy(error_estimators2[:,3], label='$\eta_{osc}$')
plt.semilogy(error_estimators2[:,4], label='$\eta_{NC}$')
plt.semilogy(error_estimators2[:,0:4].sum(axis=1),  'g^-', linewidth=2, markersize=10, label='$\eta$')
#plt.semilogy(Jup1[0:(pr[-1]+1)],  'k^-', linewidth=2, markersize=10, label='$J_u^{up}$')
#plt.semilogy(error_estimators1[:,0]*0.1,  'r-', linewidth=1.5, markersize=10, label='$\gamma_{lin}\eta_{disc}$')
plt.xlabel('Iterations', fontsize=40)
plt.ylabel('Dual error (adjoint)', fontsize=40)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(loc=0, fontsize=30)

plt.figure(3, figsize=(12,10))
plt.semilogy(error_estimators3[:,0],  'r^-', linewidth=2, markersize=10, label='$\eta_{disc}$')
plt.semilogy(error_estimators3[:,1],  'b^-', linewidth=2, markersize=10, label='$\eta_{lin}$')
plt.semilogy(error_estimators3[:,2], 'm^-', linewidth=2, markersize=10, label='$\eta_{quad}$')
plt.semilogy(error_estimators3[:,3], label='$\eta_{osc}$')
plt.semilogy(error_estimators3[:,4], label='$\eta_{NC}$')
plt.semilogy(error_estimators3[:,0:4].sum(axis=1),  'g^-', linewidth=2, markersize=10, label='$\eta$')
#plt.semilogy(Jup1[0:(pr[-1]+1)],  'k^-', linewidth=2, markersize=10, label='$J_u^{up}$')
#plt.semilogy(error_estimators1[:,0]*0.1,  'r-', linewidth=1.5, markersize=10, label='$\gamma_{lin}\eta_{disc}$')
plt.xlabel('Iterations', fontsize=40)
plt.ylabel('Dual error (f)', fontsize=40)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(loc=0, fontsize=30)