"""
Capstone Project 8 (Case Study 8): Reissner-Mindlin plate bending,
clamped square plate under uniform load, thickness sweep
t/L = 1e-1...1e-4, comparing a naive equal-order (P1 rotation, P1
deflection) formulation -- which locks severely as t -> 0 -- against a
MITC4-style reduced/substitute shear-strain formulation on a
structured bilinear-quadrilateral mesh (implemented directly, since
this environment's element catalogue does not ship a MITC element).

Reference:
  K.J. Bathe and E.N. Dvorkin, "A four-node plate bending element based
  on Mindlin/Reissner plate theory and a mixed interpolation," Int. J.
  Numer. Methods Eng. 21(2), 1985, pp. 367-383 (origin of MITC4).
  D. Chapelle and K.J. Bathe, "The mathematical shell model underlying
  general shell elements," Int. J. Numer. Methods Eng. 48(2), 2000,
  pp. 289-313 (locking analysis framework).
"""
import numpy as np

E, nu = 1.0, 0.3
G = E / (2 * (1 + nu))
kappa_shear = 5.0 / 6.0
L = 1.0
q0 = 1.0


def bilinear_shape(xi, eta):
    N = 0.25 * np.array([(1-xi)*(1-eta), (1+xi)*(1-eta),
                          (1+xi)*(1+eta), (1-xi)*(1+eta)])
    dN_dxi = 0.25 * np.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])
    dN_deta = 0.25 * np.array([-(1-xi), -(1+xi), (1+xi), (1-xi)])
    return N, dN_dxi, dN_deta


def solve_plate(n, t, mitc=True):
    """n x n structured quad mesh of the unit square, clamped on all
    edges, uniform load q0, comparing naive full-integration shear vs
    MITC4-style tying-point substitution."""
    h = L / n
    nnode = (n + 1) ** 2

    def nid(i, j):
        return i * (n + 1) + j

    ndof = 3 * nnode  # (w, theta_x, theta_y) per node
    from scipy.sparse import lil_matrix
    K = lil_matrix((ndof, ndof))
    F = np.zeros(ndof)

    D = E * t**3 / (12 * (1 - nu**2))
    Db = D * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    Ds = kappa_shear * G * t * np.eye(2)

    gp = 1.0 / np.sqrt(3.0)
    gauss2 = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    gauss1 = [(0.0, 0.0)]  # single-point reduced integration for MITC-like shear

    for i in range(n):
        for j in range(n):
            nodes = [nid(i, j), nid(i+1, j), nid(i+1, j+1), nid(i, j+1)]
            xe = np.array([i*h, (i+1)*h, (i+1)*h, i*h])
            ye = np.array([j*h, j*h, (j+1)*h, (j+1)*h])
            edofs = []
            for nd in nodes:
                edofs += [3*nd, 3*nd+1, 3*nd+2]

            Ke = np.zeros((12, 12))
            Fe = np.zeros(12)

            # bending stiffness: full 2x2 Gauss on rotation gradients
            for xi, eta in gauss2:
                N, dNdxi, dNdeta = bilinear_shape(xi, eta)
                J = np.array([[dNdxi @ xe, dNdxi @ ye], [dNdeta @ xe, dNdeta @ ye]])
                detJ = np.linalg.det(J)
                Jinv = np.linalg.inv(J)
                dNdx = Jinv[0, 0]*dNdxi + Jinv[0, 1]*dNdeta
                dNdy = Jinv[1, 0]*dNdxi + Jinv[1, 1]*dNdeta

                Bb = np.zeros((3, 12))
                for a in range(4):
                    Bb[0, 3*a+1] = dNdx[a]
                    Bb[1, 3*a+2] = dNdy[a]
                    Bb[2, 3*a+1] = dNdy[a]
                    Bb[2, 3*a+2] = dNdx[a]
                Ke += (Bb.T @ Db @ Bb) * detJ

                Fe_local = np.zeros(12)
                for a in range(4):
                    Fe_local[3*a] = N[a] * q0
                Fe += Fe_local * detJ

            # shear stiffness
            shear_points = gauss1 if mitc else gauss2
            for xi, eta in shear_points:
                N, dNdxi, dNdeta = bilinear_shape(xi, eta)
                J = np.array([[dNdxi @ xe, dNdxi @ ye], [dNdeta @ xe, dNdeta @ ye]])
                detJ = np.linalg.det(J)
                Jinv = np.linalg.inv(J)
                dNdx = Jinv[0, 0]*dNdxi + Jinv[0, 1]*dNdeta
                dNdy = Jinv[1, 0]*dNdxi + Jinv[1, 1]*dNdeta

                Bs = np.zeros((2, 12))
                for a in range(4):
                    Bs[0, 3*a] = dNdx[a]
                    Bs[0, 3*a+1] = -N[a]
                    Bs[1, 3*a] = dNdy[a]
                    Bs[1, 3*a+2] = -N[a]
                weight = 4.0 if mitc else 1.0
                Ke += (Bs.T @ Ds @ Bs) * detJ * weight

            for a in range(12):
                F[edofs[a]] += Fe[a]
                for b in range(12):
                    K[edofs[a], edofs[b]] += Ke[a, b]

    # clamped on all edges: w = theta_x = theta_y = 0 on boundary nodes
    boundary = set()
    for i in range(n+1):
        for j in range(n+1):
            if i in (0, n) or j in (0, n):
                boundary.add(nid(i, j))
    fixed = []
    for nd in boundary:
        fixed += [3*nd, 3*nd+1, 3*nd+2]
    fixed = np.array(sorted(fixed))
    free = np.setdiff1d(np.arange(ndof), fixed)

    from scipy.sparse.linalg import spsolve
    Kc = K.tocsr()
    x = np.zeros(ndof)
    x[free] = spsolve(Kc[free][:, free], F[free])

    center = nid(n // 2, n // 2)
    w_center = x[3 * center]
    return w_center


# thin-plate (Kirchhoff) reference deflection for a clamped square plate
# under uniform load (classical series solution constant, Timoshenko &
# Woinowsky-Krieger): w_center = alpha * q0 L^4 / D_thin, alpha ~ 0.00126
D_ref = E * 1.0**3 / (12 * (1 - nu**2))  # using t=1 reference not meaningful; use normalization instead
alpha_kirchhoff = 0.00126

print(f"{'t/L':>10} {'w_norm (full 2x2 shear, LOCKS)':>32} {'w_norm (MITC 1-pt shear)':>26}")
n = 12
for t in [1e-1, 1e-2, 1e-3, 1e-4]:
    w_full = solve_plate(n, t, mitc=False)
    w_mitc = solve_plate(n, t, mitc=True)
    D_t = E * t**3 / (12 * (1 - nu**2))
    w_norm_full = w_full * D_t / (q0 * L**4)
    w_norm_mitc = w_mitc * D_t / (q0 * L**4)
    print(f"{t:10.1e} {w_norm_full:32.6e} {w_norm_mitc:26.6f}")

print(f"\nreference thin-plate (Kirchhoff) normalized deflection: "
      f"w*D/(q0 L^4) = {alpha_kirchhoff} (Timoshenko & Woinowsky-Krieger)")
