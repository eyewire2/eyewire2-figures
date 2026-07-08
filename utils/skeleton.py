import numpy as np
from skeliner import Skeleton


def rotate_skel(skel, rotation_deg):
    """
    Apply rotation in the XY plane around the soma center.

    Parameters
    - skel: skeliner.core.Skeleton
        Input skeleton. Node 0 is assumed to be the soma centroid.
    - rotation_deg: float | None
        Counterclockwise rotation angle in degrees applied in the XY plane
        about the soma center. Use 0 or None for no rotation.

    Returns
    - skel_aug: skeliner.core.Skeleton
        The rotated skeleton.

    Example
    >>> skel_aug = augment_skel(skel, rotation_deg=30.0)
    """

    nodes = np.asarray(skel.nodes, dtype=np.float64).copy()
    if len(nodes) == 0:
        return skel

    # Soma center
    soma_center = np.asarray(
        skel.soma.center if skel.soma is not None else nodes[0],
        dtype=np.float64
    ).copy()

    # Build rotation matrix
    theta = np.deg2rad(rotation_deg if rotation_deg is not None else 0.0)
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    # Apply rotation around soma: x' = Rz @ (x - c) + c
    nodes = (nodes - soma_center) @ Rz.T + soma_center

    # Update soma geometry
    if skel.soma is not None:
        new_R = Rz @ skel.soma.R
        soma2 = skel.soma.__class__(
            soma_center.copy(),
            skel.soma.axes.copy(),
            new_R,
            verts=skel.soma.verts
        )
    else:
        soma2 = skel.soma

    # Build new Skeleton
    skel_aug = Skeleton(
        soma=soma2,
        nodes=nodes,
        radii={k: np.asarray(v).copy() for k, v in skel.radii.items()},
        edges=skel.edges.copy() if skel.edges is not None else None,
        ntype=np.asarray(skel.ntype).copy() if skel.ntype is not None else None,
        node2verts=skel.node2verts,
        vert2node=skel.vert2node,
        meta=dict(skel.meta) if hasattr(skel, "meta") and skel.meta is not None else {},
        extra={},
    )

    return skel_aug