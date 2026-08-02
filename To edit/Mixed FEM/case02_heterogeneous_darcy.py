"""
Capstone Project 2 (Case Study 2): Heterogeneous and anisotropic Darcy
flow, genuine RT0-DG0 mixed FEM (scikit-fem). Cellwise scalar/tensor
permeability K(x). Pressure-driven flow p=1 (x=0) / p=0 (x=1); the
top/bottom no-flow condition is imposed *strongly* on the RT flux
trial space (the essential condition of this mixed formulation --
Neumann-type data is essential here, exactly the reverse of the primal
formulation, and getting this backwards is the single most common
implementation bug in mixed Darcy codes).

Units (SI, Darcy's law u = -(K/mu) grad p):
  K [m^2] intrinsic permeability, mu [Pa.s] viscosity, p [Pa], u [m/s]
  (mu is absorbed into K here, i.e. we solve with hydraulic
  conductivity K/mu directly, consistent with the chapter text)

Reference:
  M.A. Christie and M.J. Blunt, "Tenth SPE Comparative Solution
  Project: A Comparison of Upscaling Techniques," SPE Reservoir
  Evaluation & Engineering 4(4), 2001, pp. 308-317 (motivating dataset;
  see chapter text for the real download route -- this offline
  environment cannot reach spe.org, so part (b) below uses a
  synthetic log-normal field with SPE10-like heterogeneity statistics).
  F. Brezzi, J. Douglas Jr., M. Fortin, L.D. Marini, "Efficient
  rectangular mixed finite elements in two and three space variables,"
  RAIRO Model. Math. Anal. Numer. 21(4), 1987, pp. 581-604.
"""
import numpy as np
import time
from skfem import *
from skfem.helpers import dot, div


def make_checkerboard_K(mesh, blocks=4, Klow=1.0, Khigh=1000.0):
    centers = mesh.p[:, mesh.t].mean(axis=1)
    bi = np.minimum((centers[0] * blocks).astype(int), blocks - 1)
    bj = np.minimum((centers[1] * blocks).astype(int), blocks - 1)
    return np.where((bi + bj) % 2 == 0, Khigh, Klow)


def make_spe10_like_field(mesh, corr_length=0.08, sigma=2.5, seed=1, Ngrid=128):
    rng = np.random.default_rng(seed)
    kx = np.fft.fftfreq(Ngrid).reshape(-1, 1) * Ngrid
    ky = np.fft.fftfreq(Ngrid).reshape(1, -1) * Ngrid
    kk = np.sqrt(kx**2 + ky**2)
    Sk = np.exp(-(kk * corr_length) ** 2)
    noise = rng.normal(size=(Ngrid, Ngrid)) + 1j * rng.normal(size=(Ngrid, Ngrid))
    field = np.real(np.fft.ifft2(np.fft.fft2(noise) * np.sqrt(Sk)))
    logK = sigma * (field - field.mean()) / field.std()
    centers = mesh.p[:, mesh.t].mean(axis=1)
    ix = np.clip((centers[0] * Ngrid).astype(int), 0, Ngrid - 1)
    iy = np.clip((centers[1] * Ngrid).astype(int), 0, Ngrid - 1)
    return np.exp(logK[ix, iy])


def solve_darcy(mesh, Kfield, p_left=1.0, p_right=0.0):
    e = ElementTriRT0() * ElementTriP0()
    basis = Basis(mesh, e)
    Kb = Basis(mesh, ElementTriP0())

    @BilinearForm
    def bilinf_het(sigma, u, tau, v, w):
        return (1.0 / w["Kfield"]) * dot(sigma, tau) - div(sigma) * v - div(tau) * u

    A = bilinf_het.assemble(basis, Kfield=Kb.interpolate(Kfield))

    left = mesh.facets_satisfying(lambda x: x[0] < 1e-10)
    right = mesh.facets_satisfying(lambda x: x[0] > 1 - 1e-10)
    top = mesh.facets_satisfying(lambda x: x[1] > 1 - 1e-10)
    bot = mesh.facets_satisfying(lambda x: x[1] < 1e-10)

    fb_lr = FacetBasis(mesh, e, facets=np.concatenate([left, right]))

    @LinearForm
    def bc(tau, v, w):
        pD = p_left * (w.x[0] < 1e-10) + p_right * (w.x[0] > 1 - 1e-10)
        return -dot(tau, w.n) * pD

    b = bc.assemble(fb_lr)
    dofs_noflow = basis.get_dofs(facets=np.concatenate([top, bot]))
    Acon, bcon, x0, I = condense(A, b, D=dofs_noflow)

    t0 = time.time()
    xI = solve(Acon, bcon)
    t1 = time.time()

    x = np.zeros(A.shape[0])
    x[I] = xI
    (sigma, sb), (u, ub) = basis.split(x)

    fb_left = FacetBasis(mesh, ElementTriRT0(), facets=left)
    fb_right = FacetBasis(mesh, ElementTriRT0(), facets=right)

    @Functional
    def flux_out(w):
        return dot(w["sigh"], w.n)

    qleft = flux_out.assemble(fb_left, sigh=fb_left.interpolate(sigma))
    qright = flux_out.assemble(fb_right, sigh=fb_right.interpolate(sigma))
    return sigma, u, -qleft, qright, (t1 - t0), basis.N


print("Sanity check: homogeneous K=1, expect inflow = outflow = 1.0 exactly")
mesh0 = MeshTri().refined(4)
_, _, qin0, qout0, _, _ = solve_darcy(mesh0, np.ones(mesh0.t.shape[1]))
print(f"  inflow={qin0:.10f}  outflow={qout0:.10f}\n")


print("=" * 78)
print("Part (a): checkerboard permeability, contrast sweep")
print("=" * 78)
mesh = MeshTri().refined(5)
print(f"{'contrast':>10} {'ndof':>7} {'inflow':>12} {'outflow':>12} "
      f"{'mismatch':>12} {'time(s)':>10} {'max cellwise |div sigma_h|':>28}")
for contrast in [1e0, 1e2, 1e4, 1e6]:
    Kfield = make_checkerboard_K(mesh, blocks=4, Klow=1.0, Khigh=contrast)
    sigma, u, qin, qout, dt, ndof = solve_darcy(mesh, Kfield)
    sb = Basis(mesh, ElementTriRT0())
    div_vals = div(sb.interpolate(sigma))
    max_cellwise = float(np.max(np.abs(div_vals)))
    print(f"{contrast:10.1e} {ndof:7d} {qin:12.6f} {qout:12.6f} "
          f"{abs(qin-qout):12.3e} {dt:10.4f} {max_cellwise:28.3e}")

print()
print("=" * 78)
print("Part (b): synthetic SPE10-like log-normal permeability field")
print("=" * 78)
Kfield = make_spe10_like_field(mesh)
print(f"K statistics over {len(Kfield)} elements: "
      f"min={Kfield.min():.4e}  max={Kfield.max():.4e}  "
      f"contrast={Kfield.max()/Kfield.min():.3e}")
sigma, u, qin, qout, dt, ndof = solve_darcy(mesh, Kfield)
sb = Basis(mesh, ElementTriRT0())
div_vals = div(sb.interpolate(sigma))
max_cellwise_b = float(np.max(np.abs(div_vals)))
print(f"inflow={qin:.6f}  outflow={qout:.6f}  mismatch={abs(qin-qout):.3e}  "
      f"max cellwise |div sigma_h|={max_cellwise_b:.3e}")
