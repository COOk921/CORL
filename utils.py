import pdb

import numpy as np

def _count_increasing_pairs(arr):
    """
    Count pairs (i < j) with arr[i] < arr[j].
    Uses coordinate-compression + Fenwick (BIT) -> O(len(arr) log U)
    """
    if arr.size < 2:
        return 0
    vals, inv = np.unique(arr, return_inverse=True)  # inv: 0..U-1 ranks
    U = vals.size
    bit = [0] * (U + 1)  # 1-indexed BIT

    def bit_add(i):
        i += 1
        while i <= U:
            bit[i] += 1
            i += i & -i

    def bit_sum(i):
        # sum of [0..i], i is 0-based index; if i < 0 -> 0
        if i < 0:
            return 0
        s = 0
        i += 1
        while i > 0:
            s += bit[i]
            i -= i & -i
        return s

    cnt = 0
    for rank in inv:  # iterate in original order
        if rank > 0:
            cnt += bit_sum(rank - 1)   # number of previous elements with value < current
        # else rank == 0 -> 0 previous smaller
        bit_add(rank)
    return cnt

def compute_rehandle_rate(from_layer, from_col, from_bay, from_yard, denominator='same'):
    """
    Compute '捣箱' statistics.

    Parameters
    ----------
    from_layer, from_col, from_bay, from_yard : array-like, same length N
        Positions of containers; index order is the current order.
    denominator : {'same', 'all'}
        'same' (default) -> compute rate = rehandle_pairs / (pairs sharing same yard,bay,col)
        'all' -> compute rate = rehandle_pairs / (N*(N-1)/2)

    Returns
    -------
    rehandle_pairs : int
        Number of pairs (i<j) that satisfy: same yard,bay,col AND from_layer[i] < from_layer[j]
    total_pairs_for_denominator : int
        Denominator used (either number of same-stack pairs or total possible pairs)
    rate : float
        rehandle_pairs / total_pairs_for_denominator (0.0 if denominator == 0)
    """
    # convert to numpy arrays
    from_layer = np.asarray(from_layer)
    from_col   = np.asarray(from_col)
    from_bay   = np.asarray(from_bay)
    from_yard  = np.asarray(from_yard)

    if not (from_layer.shape == from_col.shape == from_bay.shape == from_yard.shape):
        raise ValueError("All input arrays must have the same shape/length.")

    N = from_layer.size
    # build grouping key (yard, bay, col)
    keys = np.vstack((from_yard, from_bay, from_col)).T
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)

    rehandle_pairs = 0
    total_same_stack_pairs = 0
    # iterate each group (same yard,bay,col)
    for gid in range(unique_keys.shape[0]):
        idxs = np.nonzero(inverse == gid)[0]  # indices in original order (ascending)
        m = idxs.size
        total_same_stack_pairs += m * (m - 1) // 2
        if m < 2:
            continue
        layers = from_layer[idxs]
        rehandle_pairs += _count_increasing_pairs(layers)

    if denominator == 'same':
        denom = total_same_stack_pairs
    elif denominator == 'all':
        denom = N * (N - 1) // 2
    else:
        raise ValueError("denominator must be 'same' or 'all'")

    rate = (rehandle_pairs / denom) if denom > 0 else 0.0
    return int(rehandle_pairs), int(denom), float(rate)


def calculation_metrics(sequence, feature):
    # to_layer = feature[sequence, -1] 
    # to_col = feature[sequence, -2]
    # to_bay = feature[sequence, -3]

    from_layer = feature[sequence, -1]
    from_col = feature[sequence, -2]
    from_bay = feature[sequence, -3]
    from_yard = feature[sequence, -4]

    rehandle, denom, rate = compute_rehandle_rate(from_layer, from_col, from_bay, from_yard)

    print("rehandle:", rehandle)
    print("denom:", denom) 
    print("rate:", rate) # low rehandle rate is good

    return rate


# from_layer = np.array([0, 2, 1, 3, 0, 1])
# from_yard  = np.array([1, 1, 1, 1, 2, 2])
# from_bay   = np.array([1, 1, 1, 1, 1, 1])
# from_col   = np.array([1, 1, 1, 1, 1, 1])

# rehandle, denom, rate = compute_rehandle_rate(from_layer, from_col, from_bay, from_yard, denominator='same')
# print(rehandle, denom, rate) 
