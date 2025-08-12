from copy import copy, deepcopy
import sys, os
import numpy as np

def myDynamicProgramming(n, c, V, W):
    """
    0/1 Knapsack via dynamic programming.

    Returns:
        Z  : list[int]  -> chosen items (0/1 per item, original order)
        DP : list[list[int]] -> DP table of best values for capacities 0..c
    """
    # DP[i][cap] = best value using first i items with capacity 'cap'
    DP = [[0] * (c + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        wi, vi = W[i - 1], V[i - 1]
        prev = DP[i - 1]
        row = DP[i]
        for cap in range(c + 1):
            if wi > cap:
                row[cap] = prev[cap]                 # can't take item i-1
            else:
                row[cap] = max(prev[cap], prev[cap - wi] + vi)  # skip vs take

    # Reconstruct chosen items
    Z = [0] * n
    cap = c
    i = n
    while i > 0:
        if DP[i][cap] != DP[i - 1][cap]:
            Z[i - 1] = 1
            cap -= W[i - 1]
        i -= 1

    return Z, DP


# Example usage
if __name__ == "__main__":
    n = 3
    V = [5, 8, 12]
    W = [4, 5, 10]
    c = 11

    Z, DP = myDynamicProgramming(n, c, V, W)

    print("Z =", Z)
    print("DP table:")
    for row in DP:
        print(row)