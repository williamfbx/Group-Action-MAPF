import numpy as np
import scipy.sparse
import scipy.sparse.linalg


def solve_poisson(my_map, starts, goals):
    """Solve a multi-source Poisson field on the map interior.

    Implementation intentionally mirrors laplace_solver.py sparse assembly style.
    """
    maze = np.array(my_map, dtype=np.int8)
    if maze.ndim != 2:
        raise ValueError("my_map must be a 2D grid")

    # Build a one-cell obstacle perimeter around the map.
    maze = np.pad(maze, pad_width=1, mode="constant", constant_values=1)

    h, w = maze.shape
    n = h * w
    wall_mask = (maze == 1).ravel()
    flat_idx = np.arange(n).reshape(h, w)

    # Build neighbor pairs for free cells
    def neighbor(di, dj):
        si = flat_idx[max(0, di):h + min(0, di), max(0, dj):w + min(0, dj)]
        ti = flat_idx[max(0, -di):h + min(0, -di), max(0, -dj):w + min(0, -dj)]
        wall_src = (maze == 1)[max(0, di):h + min(0, di), max(0, dj):w + min(0, dj)]
        wall_tgt = (maze == 1)[max(0, -di):h + min(0, -di), max(0, -dj):w + min(0, -dj)]
        valid = ~wall_src.ravel() & ~wall_tgt.ravel()
        return si.ravel()[valid], ti.ravel()[valid]

    pairs = [neighbor(di, dj) for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]]

    # Off-diagonal entries: -1 for each free-neighbor pair
    off_r = np.concatenate([s for s, _ in pairs])
    off_c = np.concatenate([t for _, t in pairs])
    off_v = np.full(len(off_r), -1.0)

    # Diagonal: degree (number of free neighbors) for free cells, 1 for fixed
    degree = np.zeros(n)
    np.add.at(degree, off_r, 1.0)

    free_mask = ~wall_mask
    fixed = np.zeros(n, dtype=bool)

    start_indices = []
    goal_indices = []

    for start in starts:
        r, c = int(start[0]) + 1, int(start[1]) + 1
        idx = r * w + c
        if wall_mask[idx]:
            raise ValueError("Start {} lies on an obstacle".format(start))
        fixed[idx] = True
        start_indices.append(idx)

    for goal in goals:
        r, c = int(goal[0]) + 1, int(goal[1]) + 1
        idx = r * w + c
        if wall_mask[idx]:
            raise ValueError("Goal {} lies on an obstacle".format(goal))
        fixed[idx] = True
        goal_indices.append(idx)

    update = free_mask & ~fixed

    # For fixed and wall rows: identity row
    diag = np.where(update, degree, 1.0)

    # Remove off-diagonal entries belonging to fixed/wall rows
    keep = update[off_r]
    off_r, off_c, off_v = off_r[keep], off_c[keep], off_v[keep]

    diag_r = np.arange(n)
    all_r = np.concatenate([diag_r, off_r])
    all_c = np.concatenate([diag_r, off_c])
    all_v = np.concatenate([diag, off_v])

    A = scipy.sparse.csr_matrix((all_v, (all_r, all_c)), shape=(n, n))

    b = np.zeros(n)
    source_strength = 100.0
    if len(start_indices) > 0:
        b[np.array(start_indices, dtype=int)] = source_strength
    if len(goal_indices) > 0:
        b[np.array(goal_indices, dtype=int)] = -source_strength

    print("Solving sparse linear system...")
    x = scipy.sparse.linalg.spsolve(A, b)
    phi = x.reshape(h, w)
    phi[maze == 1] = 0.0

    # Return only the original map region.
    return phi[1:-1, 1:-1]


__all__ = ["solve_poisson"]