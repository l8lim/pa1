from copy import copy, deepcopy
import sys, os
import numpy as np
import itertools

def _cost_so_far(A, X):
    # Returns sum of assigned costs so far; +inf if invalid assignment
    total = 0
    for i, j in enumerate(X):
        if j != -1:
            if j < 0 or j >= A.shape[1]:
                return np.inf
            total += A[i, j]
    return total

def _remaining_rows_cols(X, n):
    rows_left = [i for i, j in enumerate(X) if j == -1]
    used = {j for j in X if j != -1}
    cols_left = [j for j in range(n) if j not in used]
    return rows_left, cols_left

def minElement(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        idx = int(np.argmin(arr))
        return arr[idx].item(), idx
    elif arr.ndim == 2:
        flat = int(np.argmin(arr))
        i, j = np.unravel_index(flat, arr.shape)
        return arr[i, j].item(), (i, j)
    else:
        raise ValueError("Only works for 1D or 2D arrays.")

def calcMatrixMinor(A, i, j):
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError("Matrix must be 2D")
    row_mask = np.ones(A.shape[0], dtype=bool)
    col_mask = np.ones(A.shape[1], dtype=bool)
    row_mask[i] = False
    col_mask[j] = False
    return A[row_mask][:, col_mask]

def rowMin(A):
    A = np.asarray(A)
    if A.size == 0:
        return 0
    return np.min(A, axis=1).sum().item()

def SNH(A):
    A = np.asarray(A)
    n = A.shape[0]
    # List all entries with costs
    all_entries = [(A[i, j].item(), i, j) for i in range(n) for j in range(n)]
    all_entries.sort(key=lambda x: x[0])

    assign = [-1] * n
    used_r, used_c = set(), set()
    total_cost = 0

    for cost, r, c in all_entries:
        if r in used_r or c in used_c:
            continue
        assign[r] = c
        used_r.add(r)
        used_c.add(c)
        total_cost += cost
        if len(used_r) == n:
            break

    if -1 in assign:
        return assign, np.inf
    return assign, total_cost

def lower_bound(A, X):
    A = np.asarray(A)
    n = A.shape[0]
    cost_now = _cost_so_far(A, X)
    if not np.isfinite(cost_now):
        return np.inf

    rows_left, cols_left = _remaining_rows_cols(X, n)
    if not rows_left:
        return cost_now
    if len(cols_left) < len(rows_left):
        return np.inf

    extra = 0
    for r in rows_left:
        vals = A[r, cols_left]
        if vals.size == 0:
            return np.inf
        extra += np.min(vals).item()
    return cost_now + extra

def upper_bound(A, X):
    A = np.asarray(A)
    n = A.shape[0]
    cost_now = _cost_so_far(A, X)
    if not np.isfinite(cost_now):
        return np.inf

    rows_left, cols_left = _remaining_rows_cols(X, n)
    if not rows_left:
        return cost_now
    if len(cols_left) < len(rows_left):
        return np.inf

    submat = A[np.array(rows_left)[:, None], np.array(cols_left)[None, :]]
    sub_assign, sub_cost = SNH(submat)
    if not np.isfinite(sub_cost):
        return np.inf
    return cost_now + sub_cost


def myBranchBound(C):
    """
    Branch-and-Bound for assignment problem.
    Returns: (best assignment as 0/1 matrix, UB history, node count)
    """
    A = np.asarray(C, dtype=int)
    n = A.shape[0]

    # Start with SNH UB
    best_assign, best_cost = SNH(A)
    ub_list = [best_cost]

    open_nodes = []
    node_count = 0
    order_id = 0

    # Helper to finish assignment from partial using SNH
    def complete_with_snh(A, X):
        X = list(X)
        rows_left, cols_left = _remaining_rows_cols(X, n)
        if not rows_left:
            return X, _cost_so_far(A, X)
        sub = A[np.array(rows_left)[:, None], np.array(cols_left)[None, :]]
        sub_assign, sub_cost = SNH(sub)
        if not np.isfinite(sub_cost):
            return X, np.inf
        for rr, jj in zip(rows_left, sub_assign):
            X[rr] = cols_left[jj]
        return X, _cost_so_far(A, X)

    # Level 0 branching
    for col in range(n):
        start_assign = [-1] * n
        start_assign[0] = col
        lb = lower_bound(A, start_assign)
        node_count += 1

        comp_assign, comp_cost = complete_with_snh(A, start_assign)
        if comp_cost < best_cost:
            best_cost = comp_cost
            best_assign = comp_assign
            ub_list.append(best_cost)

        if lb < best_cost:
            open_nodes.append([lb, order_id, 1, start_assign])
        order_id += 1

    open_nodes = [nd for nd in open_nodes if nd[0] < best_cost]

    # Main loop: pick node with smallest LB
    while open_nodes:
        idx = min(range(len(open_nodes)), key=lambda k: (open_nodes[k][0], open_nodes[k][1]))
        lb, _, level, X = open_nodes.pop(idx)

        if level == n:
            comp_cost = _cost_so_far(A, X)
            if comp_cost < best_cost:
                best_cost = comp_cost
                best_assign = X
                ub_list.append(best_cost)
                open_nodes = [nd for nd in open_nodes if nd[0] < best_cost]
            continue

        used_cols = {j for j in X if j != -1}
        for col in range(n):
            if col in used_cols:
                continue
            child_X = list(X)
            child_X[level] = col

            child_lb = lower_bound(A, child_X)
            node_count += 1

            comp_assign, comp_cost = complete_with_snh(A, child_X)
            if comp_cost < best_cost:
                best_cost = comp_cost
                best_assign = comp_assign
                ub_list.append(best_cost)
                open_nodes = [nd for nd in open_nodes if nd[0] < best_cost]

            if child_lb < best_cost:
                open_nodes.append([child_lb, order_id, level + 1, child_X])
            order_id += 1

    # Convert to 0/1 matrix
    Xmat = [[0] * n for _ in range(n)]
    for r, c in enumerate(best_assign):
        Xmat[r][c] = 1

    return Xmat, ub_list, node_count
