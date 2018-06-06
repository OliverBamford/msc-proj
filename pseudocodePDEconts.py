class PDEConstrOptProblem(object):
    def __init__(self):
        mesh = UnitSquareMesh(10,10)
        V = FunctionSpace(mesh, "CG", 1)

        X = SpatialCoordinate(mesh)
        u_t = sin(X[0])*cos(X[1])
        alpha = 0.3

        self.m = Function(V)
        self.dm = Function(V)
        self.u = Function(V)
        self.lamb = Function(V)

        v = TestFunction(V)

        #state equation
        u_ = TrialFunction(V)
        self.F = inner(grad(u_), grad(v))*dx - self.m*v*dx
        self.bc = Dirichlet(V, 0., "on_boundary")
        #form of misfit functional
        self.J_form = 0.5*((self.u - u_t)**2*dx + self.alpha*self.m**2*dx)
        #form of adjoint
        lambd_ = TrialFunction(V)
        self.F_adj = inner(grad(v), grad(lamb_))*dx + (self.u - u_t)

    def solve_state(self):
        solve(self.F == 0, self.u, self.bc)

    def eval_J(self):
        self.solve_state()
        return assemble(self.J_form)

    def solve_adjoint(self):
        solve(self.F_adj == 0, self.lamb, self.bc)

    def compute_RieszRep(self):
        self.solve_state()
        self.solve_adjoint()
        v = TestFunction(V)
        dJ(v) = (self.alpha*self.m*v - self.lamb*v)*dx
        F_R = self.dm * v * dx + dJ(v)
        solve(F_R == 0, self.dm)

    def update_control(self, step)
        self.m += step * self.dm


P = PDEConstrOptProblem()

for ii in range(5):
    Jval[ii] = eval_J()
    compute_RieszRep()

    for jj in range(3):
        update_control(0.1)
        Jtemp[jj] = eval_J()

    minIdx = indexOfMinimum(Jtemp)
    update_control(-0.3+0.1*(1+minIdx))

