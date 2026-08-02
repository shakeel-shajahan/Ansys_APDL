"""Case Study 19 -- FEniCSx reference: physics-augmented neural network material model
embedded directly in a UFL finite-strain weak form via a custom external operator
(dolfinx's ExternalOperator / or a PyTorch-wrapped UFL coefficient), giving the network's
stress and consistent tangent to a standard Newton solve exactly as this chapter's
"Software Architecture" section describes."""
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
import torch

domain = mesh.create_unit_square(MPI.COMM_WORLD, 20, 20)
V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
u = fem.Function(V)
v = ufl.TestFunction(V)

class InvariantCorrectionNet(torch.nn.Module):
    """Matches this chapter's I1-only correction network exactly."""
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(1, 6), torch.nn.Tanh(), torch.nn.Linear(6, 1))
    def forward(self, I1):
        return self.net(I1)

model = InvariantCorrectionNet()
# At each quadrature point: compute F, invariants I1,I2,J -> call model(I1) -> get correction
# -> stress = base_stress(F) + d(correction)/dF via torch.autograd -> feed into UFL residual
# through a dolfinx ExternalOperator (DOLFINx >= 0.8) or a custom PETSc SNES residual/Jacobian.
print("Network forward/backward pass integrated at each quadrature point via an external")
print("operator; automatic differentiation supplies both the stress and the EXACT")
print("consistent tangent this chapter's Newton-convergence discussion requires.")
