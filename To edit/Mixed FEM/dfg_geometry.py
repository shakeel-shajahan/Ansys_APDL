"""
Generate the DFG 2D-1/2D-2 benchmark geometry (Schafer & Turek, 1996):
channel [0, 2.2] x [0, 0.41], cylinder of diameter D=0.1 centered at
(0.2, 0.2) (slightly off-center from the channel mid-height, exactly as
specified in the benchmark to avoid a spurious symmetric solution).
"""
import gmsh
import sys

def build_mesh(filename="dfg_cylinder.msh", lc_far=0.02, lc_cyl=0.006):
    gmsh.initialize()
    gmsh.model.add("dfg_cylinder")

    L, H = 2.2, 0.41
    cx, cy, r = 0.2, 0.2, 0.05

    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc_far)
    p2 = gmsh.model.geo.addPoint(L, 0, 0, lc_far)
    p3 = gmsh.model.geo.addPoint(L, H, 0, lc_far)
    p4 = gmsh.model.geo.addPoint(0, H, 0, lc_far)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    pc = gmsh.model.geo.addPoint(cx, cy, 0, lc_cyl)
    pc1 = gmsh.model.geo.addPoint(cx + r, cy, 0, lc_cyl)
    pc2 = gmsh.model.geo.addPoint(cx, cy + r, 0, lc_cyl)
    pc3 = gmsh.model.geo.addPoint(cx - r, cy, 0, lc_cyl)
    pc4 = gmsh.model.geo.addPoint(cx, cy - r, 0, lc_cyl)

    ca1 = gmsh.model.geo.addCircleArc(pc1, pc, pc2)
    ca2 = gmsh.model.geo.addCircleArc(pc2, pc, pc3)
    ca3 = gmsh.model.geo.addCircleArc(pc3, pc, pc4)
    ca4 = gmsh.model.geo.addCircleArc(pc4, pc, pc1)

    outer = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    inner = gmsh.model.geo.addCurveLoop([ca1, ca2, ca3, ca4])
    surf = gmsh.model.geo.addPlaneSurface([outer, inner])

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [l4], name="inlet")
    gmsh.model.addPhysicalGroup(1, [l2], name="outlet")
    gmsh.model.addPhysicalGroup(1, [l1, l3], name="walls")
    gmsh.model.addPhysicalGroup(1, [ca1, ca2, ca3, ca4], name="cylinder")
    gmsh.model.addPhysicalGroup(2, [surf], name="fluid")

    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    gmsh.finalize()
    return n_nodes


if __name__ == "__main__":
    n = build_mesh()
    print(f"Mesh written with {n} nodes.")
