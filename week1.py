#Based on code from the book 'Solving PDEs in Python: The FEniCS Tutorial I'

from fenics import *
# Create mesh and define function space
mesh = UnitSquareMesh(100, 100)
V = FunctionSpace(mesh, 'P', 2)
f = Expression('cos(20*x[0])*sin(15*x[1])', degree=2)

#'One shot' method

# Define variational problem
u = TrialFunction(V)
phi = TestFunction(V)

a = u*phi*dx
L = f*phi*dx

# Compute solution
u = Function(V)
solve(a == L, u)

# Plot solution and mesh
plot(u)

# Compute error in L2 norm
error_L2 = errornorm(u_D, u, 'L2')

# Compute maximum error at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)
import numpy as np
error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# Print errors
print('error_L2  =', error_L2)
print('error_max =', error_max)

#Iterative method

un = 0

import numpy as np
F = np.zeros(10)
 
for i in range(0,10):
    
    #find grad(F) using current u iterate
    GF = TrialFunction(V)
    a = GF*phi*dx
    L = (un - f)*phi*dx
    
    GF = Function(V)
    solve(a == L, GF)
    
    #calculate new un iterate 
    #(must use assign method to ensure un has dolfin.functions.function.Function type)
    GF.assign(un - 0.25 * GF)
    un = GF
    F[i] = errornorm(u_D, un, 'L2')
    
import matplotlib.pyplot as plt
plt.figure()
plt.plot(F)
    
    