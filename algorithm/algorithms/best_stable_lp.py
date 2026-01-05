from __future__ import annotations

from typing import Dict, List, Tuple, Optional


def best_stable_lp(
    proposers: Dict[str, List[str]],
    receivers: Dict[str, List[str]],
    weights: Dict[Tuple[str, str], float],
    status_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Maximum-weight stable matching via a simple MILP over the
    stable-matching polytope for strict bipartite preferences.
    """
    def _fallback() -> Dict[str, str]:
        from matching.da import gale_shapley_da

        return gale_shapley_da(proposers, receivers)

    try:
        import pulp  # type: ignore
    except Exception:
        # fall back to DA if MILP is unavailable
        if status_path:
            try:
                with open(status_path, "a", encoding="utf-8") as f:
                    f.write("best_stable_lp: pulp_not_available\n")
            except Exception:
                pass
        return _fallback()

    left = list(proposers.keys())
    right = list(receivers.keys())

    # allowed edges: present in both lists
    edges: List[Tuple[str, str]] = []
    for i in left:
        for j in proposers[i]:
            if j in receivers and i in receivers[j]:
                edges.append((i, j))

    # rankings for stability constraints
    rank_p: Dict[str, Dict[str, int]] = {
        i: {j: idx for idx, j in enumerate(lst)} for i, lst in proposers.items()
    }
    rank_r: Dict[str, Dict[str, int]] = {
        j: {i: idx for idx, i in enumerate(lst)} for j, lst in receivers.items()
    }

    prob = pulp.LpProblem("best_stable_matching", pulp.LpMaximize)

    x: Dict[Tuple[str, str], pulp.LpVariable] = {}
    for i, j in edges:
        x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1, cat="Binary")

    # capacity constraints
    for i in left:
        prob += (
            pulp.lpSum(x[(i, j)] for (ii, j) in edges if ii == i) <= 1,
            f"cap_left_{i}",
        )
    for j in right:
        prob += (
            pulp.lpSum(x[(i, j)] for (i, jj) in edges if jj == j) <= 1,
            f"cap_right_{j}",
        )

    # stability constraints
    for i, j in edges:
        better_i = [k for k in proposers[i] if rank_p[i].get(k, 10**9) < rank_p[i][j]]
        better_j = [h for h in receivers[j] if rank_r[j].get(h, 10**9) < rank_r[j][i]]
        lhs_left = pulp.lpSum(x[(i, k)] for k in better_i if (i, k) in x)
        lhs_right = pulp.lpSum(x[(h, j)] for h in better_j if (h, j) in x)
        prob += (
            lhs_left + lhs_right + x[(i, j)] >= 1,
            f"stab_{i}_{j}",
        )

    prob += pulp.lpSum(weights.get((i, j), 0.0) * x[(i, j)] for i, j in edges)

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status_str = pulp.LpStatus.get(prob.status, "unknown")
    if status_path:
        try:
            with open(status_path, "a", encoding="utf-8") as f:
                f.write(f"best_stable_lp: {status_str}\n")
        except Exception:
            pass
    if status_str != "Optimal":
        return _fallback()

    match: Dict[str, str] = {}
    for i, j in edges:
        var = x[(i, j)]
        if var.value() is not None and var.value() > 0.5:
            match[i] = j
            match[j] = i
    return match
