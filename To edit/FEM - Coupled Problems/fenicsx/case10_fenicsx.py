"""Case Study 10 -- FEniCSx reference: full-order alternative to the reduced-order wake
oscillator, i.e. genuine unsteady Navier-Stokes around an elastically-mounted cylinder,
coupled to a 2-DOF rigid-body ODE (mass-spring-damper) via preCICE or a monolithic ALE
formulation. Included for completeness; the reduced-order model remains this book's
primary (and, for design-space sweeps, preferred) tool -- see this chapter's "Why This
Numerical Method" discussion."""
import ufl
from dolfinx import mesh
from mpi4py import MPI

domain = mesh.create_rectangle(MPI.COMM_WORLD, [[-5,-5],[15,5]], [200,100])
print("Full-order VIV requires a body-fitted or immersed cylinder mesh with ALE motion")
print("tied to a rigid-body ODE for the cylinder's cross-flow displacement -- substantially")
print("more expensive than the validated reduced-order wake-oscillator model used in this")
print("chapter's main code, and only worth the cost for final, single-point verification.")
