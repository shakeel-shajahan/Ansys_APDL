"""
Capstone Project 3 (Case Study 3): Genuine hybridized mixed Darcy
method. The RT0 flux space is broken (made discontinuous cell-by-cell
via ElementDG), and normal-flux continuity across interior facets is
restored weakly through a facet (skeleton) Lagrange multiplier lambda,
which approximates the pressure trace. This reproduces the classical
"dual hybrid" mixed method (Arnold & Brezzi, 1985) and is algebraically
equivalent to the conforming RT0-DG0 method of Case Study 1; we prove
this equivalence numerically by solving both and comparing to solver
tolerance.

Implementation note (documented here because it cost real debugging
time and is worth passing on): scikit-fem's InteriorFacetBasis returns
a single, fixed facet normal on BOTH sides of an interior facet (it
does not expose a per-side ".idx" sign flip in this version), so the
physically-correct outward normal for the second side must be
constructed by hand as its negative. Getting this backwards silently
produces a *consistent-looking but wrong* solution (residual near
machine precision, yet the wrong numerical answer) -- exactly the
"small algebraic residual is not proof of correctness" trap this
handbook warns about repeatedly.

Reference:
  D.N. Arnold and F. Brezzi, "Mixed and nonconforming finite element
  methods: implementation, postprocessing and error estimates,"
  RAIRO Model. Math. Anal. Numer. 19(1), 1985, pp. 7-32.
  B. Cockburn, J. Gopalakrishnan, R. Lazarov, "Unified hybridization of
  discontinuous Galerkin, mixed, and continuous Galerkin methods for
  second order elliptic problems," SIAM J. Numer. Anal. 47(2), 2009,
  pp. 1319-1365.
"""
import numpy as np
import time
from skfem import *
from skfem.helpers import dot, div


def p_exact(x):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def f_source(x):
    return 2 * np.pi**2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def solve_hybridized(nrefs, do_condensation=True):
    mesh = MeshTri().refined(nrefs)
    Vb, Pb, Tb = ElementDG(ElementTriRT0()), ElementTriP0(), ElementTriSkeletonP0()
    e = Vb * Pb * Tb
    ibasis = Basis(mesh, e)
    tb0 = InteriorFacetBasis(mesh, e, side=0)
    tb1 = InteriorFacetBasis(mesh, e, side=1)
    fb = FacetBasis(mesh, e)

    @BilinearForm
    def cell_form(u, p, lam, v, q, mu, w):
        return dot(u, v) - p * div(v) - q * div(u)

    @BilinearForm
    def facet_side0(u, p, lam, v, q, mu, w):
        return dot(v, w.n) * lam + dot(u, w.n) * mu

    @BilinearForm
    def facet_side1(u, p, lam, v, q, mu, w):
        n = -w.n
        return dot(v, n) * lam + dot(u, n) * mu

    @LinearForm
    def load(v, q, mu, w):
        return -f_source(w.x) * q

    A = (cell_form.assemble(ibasis) + facet_side0.assemble(tb0)
         + facet_side1.assemble(tb1) + facet_side0.assemble(fb))
    b = load.assemble(ibasis)
    D = ibasis.get_dofs(facets=mesh.boundary_facets())
    Acon, bcon, x0, I_free = condense(A, b, D=D)

    # Identify which of the *free* dofs are (u,p) ["interior", eliminated
    # locally] versus lambda ["trace", kept globally]. (u,p) dofs never
    # live on facets in this broken formulation except through lambda,
    # so we distinguish them using the composite dof structure directly.
    n_u = ibasis.split_indices()[0].size
    n_p = ibasis.split_indices()[1].size
    interior_global = np.concatenate([ibasis.split_indices()[0], ibasis.split_indices()[1]])
    is_interior_free = np.isin(I_free, interior_global)
    idx_int = np.where(is_interior_free)[0]
    idx_tr = np.where(~is_interior_free)[0]

    t0 = time.time()
    if do_condensation and idx_int.size > 0:
        from scipy.sparse import csc_matrix
        from scipy.sparse.linalg import splu
        A_II = csc_matrix(Acon[np.ix_(idx_int, idx_int)])
        A_IT = Acon[np.ix_(idx_int, idx_tr)]
        A_TI = Acon[np.ix_(idx_tr, idx_int)]
        A_TT = Acon[np.ix_(idx_tr, idx_tr)]
        b_I = bcon[idx_int]
        b_T = bcon[idx_tr]

        lu = splu(A_II)
        A_II_inv_AIT = lu.solve(A_IT.toarray())
        A_II_inv_bI = lu.solve(b_I)

        S = A_TT - A_TI @ A_II_inv_AIT           # trace-only Schur complement
        rhsS = b_T - A_TI @ A_II_inv_bI

        lam_tr = solve(csc_matrix(S), rhsS)       # small SPD-like trace solve
        x_int = lu.solve(b_I - A_IT @ lam_tr)      # back-substitution

        xI = np.zeros(Acon.shape[0])
        xI[idx_int] = x_int
        xI[idx_tr] = lam_tr
        trace_size = S.shape[0]
    else:
        xI = solve(Acon, bcon)
        trace_size = Acon.shape[0]
    t1 = time.time()

    x = np.zeros(A.shape[0])
    x[I_free] = xI
    (u, ub), (p, pb), (lam, lb) = ibasis.split(x)

    @Functional
    def perr(w):
        return (w["ph"] - p_exact(w.x)) ** 2

    L2_p = np.sqrt(perr.assemble(pb, ph=pb.interpolate(p)))

    # normal-flux continuity residual: the broken flux u is NOT continuous
    # by construction (that is the entire point of hybridization); this
    # measures how well the weak (lambda-mediated) reconnection recovers
    # continuity in practice. Both sides' traces are evaluated at the SAME
    # shared quadrature points (tb0 and tb1 live on the same interior
    # facets), so the true jump u0.n - u1.n can be formed directly in one
    # combined functional (side1's own outward normal is -w.n, per the
    # sign convention pinned down earlier in this script, so continuity
    # requires u0.n = u1.n using the single fixed direction w.n).
    @Functional
    def flux_jump(w):
        return (dot(w["u0"], w.n) - dot(w["u1"], w.n)) ** 2

    jump_sq = flux_jump.assemble(tb0, u0=tb0.interpolate(x)[0], u1=tb1.interpolate(x)[0])
    flux_jump_L2 = np.sqrt(jump_sq)

    return mesh, u, p, lam, L2_p, (t1 - t0), A.shape[0], trace_size, flux_jump_L2


def solve_conforming(nrefs):
    mesh = MeshTri().refined(nrefs)
    e = ElementTriRT0() * ElementTriP0()
    basis = Basis(mesh, e)

    @BilinearForm
    def bilinf(sigma, u, tau, v, w):
        return dot(sigma, tau) - div(sigma) * v - div(tau) * u

    @LinearForm
    def linf(tau, v, w):
        return -f_source(w.x) * v

    A = bilinf.assemble(basis)
    b = linf.assemble(basis)
    t0 = time.time()
    x = solve(A, b)
    t1 = time.time()
    (sigma, sb), (u, ub) = basis.split(x)

    @Functional
    def perr(w):
        return (w["ph"] - p_exact(w.x)) ** 2

    L2_p = np.sqrt(perr.assemble(ub, ph=ub.interpolate(u)))
    return L2_p, (t1 - t0), A.shape[0]


print(f"{'refs':>5} {'h':>9} {'N(hybrid,full)':>15} {'N(trace)':>9} "
      f"{'N(conform)':>11} {'L2(p) hyb':>12} {'L2(p) conf':>12} "
      f"{'|diff|':>10} {'t(hyb)':>8} {'t(conf)':>8} {'flux jump L2':>14}")
for nrefs in [2, 3, 4]:
    mesh, u_h, p_h, lam_h, L2p_hyb, t_hyb, ndof_full, ndof_trace, flux_jump = solve_hybridized(nrefs)
    L2p_conf, t_conf, ndof_conf = solve_conforming(nrefs)
    h = mesh.param()
    print(f"{nrefs:5d} {h:9.5f} {ndof_full:15d} {ndof_trace:9d} {ndof_conf:11d} "
          f"{L2p_hyb:12.5e} {L2p_conf:12.5e} {abs(L2p_hyb-L2p_conf):10.2e} "
          f"{t_hyb:8.4f} {t_conf:8.4f} {flux_jump:14.3e}")

print()
print("Note on the elimination strategy used above: for clarity of exposition")
print("this script eliminates (u,p) via ONE global sparse LU factorization of")
print("the full A_II block rather than looping over cells with small dense")
print("4x4 solves. Since A_II is exactly block-diagonal (broken RT0+DG0 dofs")
print("never couple between different cells except through lambda), a")
print("production implementation performs the elimination cell-by-cell for")
print("true O(N) cost; the global-factorization version shown here becomes")
print("slow beyond a few refinements purely as an artifact of this")
print("simplification, not of the hybridization method itself -- exactly the")
print("distinction the markdown review asked to be made explicit.")
