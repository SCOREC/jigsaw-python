"""
* DEMO-0 --- a simple example demonstrating the construction
*   of 2-d. geometry and user-defined mesh-size constraints.
*
* These examples call to JIGSAW via its api.-lib. interface.
*
* Writes "case_0x.vtk" files on output for vis. in PARAVIEW.
*
"""

import os
import numpy as np
import jigsawpy


def case_0a(src_path, dst_path):

    opts = jigsawpy.jigsaw_jig_t()

    geom = jigsawpy.jigsaw_msh_t()
    mesh = jigsawpy.jigsaw_msh_t()

#------------------------------------ define JIGSAW geometry

    geom.mshID = "euclidean-mesh"
    geom.ndims = +2
    geom.vert2 = np.array([   # list of xy "node" coordinate
        ((0, 0), 0),          # outer square
        ((9, 0), 0),
        ((9, 9), 0),
        ((0, 9), 0),
        ((4, 4), 1),          # inner square
        ((5, 4), 1),
        ((5, 5), 1),
        ((4, 5), 1)],
        dtype=geom.VERT2_t)

    geom.edge2 = np.array([   # list of "edges" between vert
        ((0, 1), 0),          # outer square
        ((1, 2), 0),
        ((2, 3), 0),
        ((3, 0), 0),
        ((4, 5), 2),          # inner square
        ((5, 6), 2),
        ((6, 7), 2),
        ((7, 4), 2)],
        dtype=geom.EDGE2_t)

#------------------------------------ build mesh via JIGSAW!

    print("Call libJIGSAW: case 0a.")

    opts.hfun_hmax = 0.05               # push HFUN limits

    opts.mesh_dims = +2                 # 2-dim. simplexes

    opts.optm_qlim = +.95

    opts.mesh_top1 = True               # for sharp feat's
    opts.geom_feat = True

    jigsawpy.lib.jigsaw(opts, geom, mesh)

    scr2 = jigsawpy.triscr2(            # "quality" metric
        mesh.point["coord"],
        mesh.tria3["index"])

    print("Saving case_0a.vtk file.")

    jigsawpy.savemsh(os.path.join(
        dst_path, "case_0a-MESH.msh"), mesh)

    # It seems that there is no single list of edges in jigsawpy that is all 
    # the edges in the mesh.  mesh.edge2 are edges 'classified' on the geometric
    # model boundary.  mesh.tria3 (and other cell types) appear to simply store
    # their bounding vertices instead of bounding edges (or dim-1 entities).
    # The edges that bound interior triangles may not exist in the data
    # structure.  Retrieving classification may be difficult.
    jigsawpy.savevtk1d(os.path.join(
        dst_path, "case_0a-1d.vtk"), mesh)

    return


def case_0b(src_path, dst_path):

    opts = jigsawpy.jigsaw_jig_t()

    geom = jigsawpy.jigsaw_msh_t()
    mesh = jigsawpy.jigsaw_msh_t()

#------------------------------------ define JIGSAW geometry

    geom.mshID = "euclidean-mesh"
    geom.ndims = +2
    geom.vert2 = np.array([   # list of xy "node" coordinate
        ((0, 0), 0),            # outer square
        ((9, 0), 0),
        ((9, 9), 0),
        ((0, 9), 0),
        ((2, 2), 0),            # inner square
        ((7, 2), 0),
        ((7, 7), 0),
        ((2, 7), 0),
        ((3, 3), 0),
        ((6, 6), 0)],
        dtype=geom.VERT2_t)

    geom.edge2 = np.array([   # list of "edges" between vert
        ((0, 1), 0),            # outer square
        ((1, 2), 0),
        ((2, 3), 0),
        ((3, 0), 0),
        ((4, 5), 0),            # inner square
        ((5, 6), 0),
        ((6, 7), 0),
        ((7, 4), 0),
        ((8, 9), 0)],
        dtype=geom.EDGE2_t)

    et = jigsawpy.\
        jigsaw_def_t.JIGSAW_EDGE2_TAG

    geom.bound = np.array([
        (1, 0, et),
        (1, 1, et),
        (1, 2, et),
        (1, 3, et),
        (1, 4, et),
        (1, 5, et),
        (1, 6, et),
        (1, 7, et),
        (2, 4, et),
        (2, 5, et),
        (2, 6, et),
        (2, 7, et)],
        dtype=geom.BOUND_t)

#------------------------------------ build mesh via JIGSAW!

    print("Call libJIGSAW: case 0b.")

    opts.hfun_hmax = 0.05               # push HFUN limits

    opts.mesh_dims = +2                 # 2-dim. simplexes

    opts.optm_qlim = +.95

    opts.mesh_top1 = True               # for sharp feat's
    opts.geom_feat = True

    jigsawpy.savemsh(os.path.join(
        dst_path, "case_0b.msh"), mesh)

    jigsawpy.lib.jigsaw(opts, geom, mesh)

    jigsawpy.savemsh(os.path.join(
        dst_path, "case_0b-MESH.msh"), mesh)

    print("Saving case_0b.vtk file.")

    jigsawpy.savevtk(os.path.join(
        dst_path, "case_0b.vtk"), mesh)

    return


def case_0c(src_path, dst_path):

    opts = jigsawpy.jigsaw_jig_t()

    geom = jigsawpy.jigsaw_msh_t()
    hmat = jigsawpy.jigsaw_msh_t()
    mesh = jigsawpy.jigsaw_msh_t()

#------------------------------------ define JIGSAW geometry

    geom.mshID = "euclidean-mesh"
    geom.ndims = +2
    geom.vert2 = np.array([   # list of xy "node" coordinate
        ((0, 0), 0),          # outer square
        ((9, 0), 0),
        ((9, 9), 0),
        ((0, 9), 0),
        ((4, 4), 0),          # inner square
        ((5, 4), 0),
        ((5, 5), 0),
        ((4, 5), 0)],
        dtype=geom.VERT2_t)

    geom.edge2 = np.array([   # list of "edges" between vert
        ((0, 1), 0),          # outer square
        ((1, 2), 0),
        ((2, 3), 0),
        ((3, 0), 0),
        ((4, 5), 0),          # inner square
        ((5, 6), 0),
        ((6, 7), 0),
        ((7, 4), 0)],
        dtype=geom.EDGE2_t)

#------------------------------------ compute HFUN over GEOM

    xgeo = geom.vert2["coord"][:, 0]
    ygeo = geom.vert2["coord"][:, 1]

    xpos = np.linspace(
        xgeo.min(), xgeo.max(), 32)

    ypos = np.linspace(
        ygeo.min(), ygeo.max(), 16)

    xmat, ymat = np.meshgrid(xpos, ypos)

    hfun = -0.4 * np.exp(-(
        0.1 * (xmat - 4.5) ** 2 +
        0.1 * (ymat - 4.5) ** 2)) + 0.6

    hmat.mshID = "euclidean-grid"
    hmat.ndims = +2
    hmat.xgrid = np.array(
        xpos, dtype=hmat.REALS_t)
    hmat.ygrid = np.array(
        ypos, dtype=hmat.REALS_t)
    hmat.value = np.array(
        hfun, dtype=hmat.REALS_t)

#------------------------------------ build mesh via JIGSAW!

    print("Call libJIGSAW: case 0c.")

    opts.hfun_scal = "absolute"
    opts.hfun_hmax = float("inf")       # null HFUN limits
    opts.hfun_hmin = float(+0.00)

    opts.mesh_dims = +2                 # 2-dim. simplexes

    opts.optm_qlim = +.95

    opts.mesh_top1 = True               # for sharp feat's
    opts.geom_feat = True

    jigsawpy.lib.jigsaw(opts, geom, mesh,
                        None, hmat)

    print("Saving case_0c.vtk file.")

    jigsawpy.savevtk(os.path.join(
        dst_path, "case_0c.vtk"), mesh)

    return

def case_0d(src_path, dst_path):

    opts = jigsawpy.jigsaw_jig_t()

    geom = jigsawpy.jigsaw_msh_t()
    mesh = jigsawpy.jigsaw_msh_t()

#------------------------------------ define JIGSAW geometry

    geom.mshID = "euclidean-mesh"
    geom.ndims = +2
    geom.vert2 = np.array([   # list of xy "node" coordinate
        ((0, 0), 0),          # outer square 
        ((9, 0), 0),
        ((9, 9), 0),
        ((0, 9), 0),
        ((2, 4), 1),          # horizontal edge
        ((7, 4), 1),
        ((4, 2), 2),          # vertical edge
        ((4, 7), 2),
        ((4, 4), 3)],         # intersection point
        dtype=geom.VERT2_t)

    geom.edge2 = np.array([   # list of "edges" between vert
        ((0, 1), 0),          # outer square
        ((1, 2), 0),
        ((2, 3), 0),
        ((3, 0), 0),
        ((4, 8), 4),          # horizontal edge - left
        ((8, 5), 5),          # horizontal edge - right
        ((6, 8), 6),          # vertical edge - bottom
        ((8, 7), 7)],         # vertical edge - top
        dtype=geom.EDGE2_t)

    et = jigsawpy.\
        jigsaw_def_t.JIGSAW_EDGE2_TAG

    # id, object index, object type
    geom.bound = np.array([
        (1, 0, et),
        (1, 1, et),
        (1, 2, et),
        (1, 3, et)],
        dtype=geom.BOUND_t)

#------------------------------------ build mesh via JIGSAW!

    print("Call libJIGSAW: case 0d.")

    opts.hfun_hmax = 0.05               # push HFUN limits

    opts.mesh_dims = +2                 # 2-dim. simplexes

    opts.optm_qlim = +.95

    opts.mesh_top1 = True               # for sharp feat's
    opts.geom_feat = True

    jigsawpy.lib.jigsaw(opts, geom, mesh)

    scr2 = jigsawpy.triscr2(            # "quality" metric
        mesh.point["coord"],
        mesh.tria3["index"])

    print("Saving case_0d.vtk file.")

    jigsawpy.savemsh(os.path.join(
        dst_path, "case_0d-MESH.msh"), mesh)

    jigsawpy.savevtk1d(os.path.join(
        dst_path, "case_0d-1d.vtk"), mesh)

    jigsawpy.savevtk(os.path.join(
        dst_path, "case_0d.vtk"), mesh)

    return


def case_0e(src_path, dst_path):

    opts = jigsawpy.jigsaw_jig_t()

    geom = jigsawpy.jigsaw_msh_t()
    mesh = jigsawpy.jigsaw_msh_t()

#------------------------------------ define JIGSAW geometry

    geom.mshID = "euclidean-mesh"
    geom.ndims = +2
    geom.vert2 = np.array([   # list of xy "node" coordinate
        ((0, 0), 0),            # outer square
        ((9, 0), 0),
        ((9, 9), 0),
        ((0, 9), 0),
        ((2, 2), 0),            # inner square
        ((7, 2), 0),
        ((7, 7), 0),
        ((2, 7), 0),
        ((3, 3), 0),            # innermost square
        ((6, 3), 0),
        ((6, 6), 0),
        ((3, 6), 0)],
        dtype=geom.VERT2_t)

    geom.edge2 = np.array([   # list of "edges" between vert
        ((0, 1), 0),            # outer square
        ((1, 2), 0),
        ((2, 3), 0),
        ((3, 0), 0),
        ((4, 5), 1),            # inner square
        ((5, 6), 1),
        ((6, 7), 1),
        ((7, 4), 1),
        ((8, 9), 2),            # innermost square
        ((9, 10), 2),
        ((10, 11), 2),
        ((11, 8), 2)],
        dtype=geom.EDGE2_t)

    et = jigsawpy.\
        jigsaw_def_t.JIGSAW_EDGE2_TAG

    geom.bound = np.array([
        (1, 0, et),  # outer
        (1, 1, et),
        (1, 2, et),
        (1, 3, et)],
        dtype=geom.BOUND_t)

#------------------------------------ build mesh via JIGSAW!

    print("Call libJIGSAW: case 0e.")

    opts.hfun_hmax = 0.05               # push HFUN limits

    opts.mesh_dims = +2                 # 2-dim. simplexes

    opts.optm_qlim = +.95

    opts.mesh_top1 = True               # for sharp feat's
    opts.geom_feat = True

    jigsawpy.savemsh(os.path.join(
        dst_path, "case_0e.msh"), mesh)

    jigsawpy.lib.jigsaw(opts, geom, mesh)

    jigsawpy.savemsh(os.path.join(
        dst_path, "case_0e-MESH.msh"), mesh)

    print("Saving case_0e.vtk file.")

    jigsawpy.savevtk(os.path.join(
        dst_path, "case_0e.vtk"), mesh)

    jigsawpy.savevtk1d(os.path.join(
        dst_path, "case_0e-1d.vtk"), mesh)

    return




def case_0_(src_path, dst_path):

#------------------------------------ build various examples

    case_0a(src_path, dst_path)
    case_0b(src_path, dst_path)
    case_0c(src_path, dst_path)
    case_0d(src_path, dst_path)
    case_0e(src_path, dst_path)

    return
