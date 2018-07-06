from fenics import *
import numpy as np

mesh = UnitSquareMesh(1,1)
V = VectorFunctionSpace(mesh, "DG", 1)
u = interpolate(Expression(['x[0]','x[1]'], degree=1), V)
u_ = np.zeros(u.vector().get_local().shape)

dm = V.dofmap()
ii = 0.
for cell in cells(mesh):
    ub = u.vector().get_local()[dm.cell_dofs(cell.index())]
    mp = cell.midpoint().array()
    mp[0] += ii
    mp[1] += ii
    ub = 2*ub - np.array([mp[0], mp[0], mp[0], mp[1], mp[1], mp[1]])
    u_[dm.cell_dofs(cell.index())] = ub
    ii += 1

u.vector().set_local(u_)
out = File("u.pvd")
out << u
