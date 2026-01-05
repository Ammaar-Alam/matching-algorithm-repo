from __future__ import annotations

from typing import Dict, List, Optional


def _delete_pair(prefs: Dict[str, List[str]], x: str, y: str) -> None:
    lst = prefs.get(x)
    if lst is None:
        return
    try:
        lst.remove(y)
    except ValueError:
        return


def irving_strict(prefs_in: Dict[str, List[str]]) -> Optional[Dict[str, str]]:
    """
    Irving's algorithm for strict stable roommates.
    prefs_in maps person -> strict list of partners.
    Returns a stable matching dict or None if none exists.
    """
    prefs: Dict[str, List[str]] = {k: list(v) for k, v in prefs_in.items()}
    # phase 1
    proposals: Dict[str, str] = {}
    for x in prefs.keys():
        proposals[x] = prefs[x][0] if prefs[x] else None  # type: ignore
    while True:
        changed = False
        for x, lst in list(prefs.items()):
            if not lst:
                return None
            y = lst[0]
            y_list = prefs.get(y, [])
            if x not in y_list:
                _delete_pair(prefs, x, y)
                changed = True
                continue
            # truncate y's list after x
            idx = y_list.index(x)
            for z in y_list[idx + 1 :]:
                _delete_pair(prefs, z, y)
                _delete_pair(prefs, y, z)
                changed = True
        if not changed:
            break
    # phase 2
    # build list of men with more than one option
    while True:
        x = next((p for p, lst in prefs.items() if len(lst) > 1), None)
        if x is None:
            break
        y = prefs[x][1]
        y_list = prefs.get(y, [])
        if not y_list:
            return None
        z = y_list[-1]
        while True:
            _delete_pair(prefs, z, prefs[z][0])
            _delete_pair(prefs, prefs[z][0], z)
            if not prefs[z]:
                return None
            if prefs[z][0] == x:
                break
            z = prefs[z][0]
    # construct matching
    match: Dict[str, str] = {}
    for x, lst in prefs.items():
        if not lst:
            return None
        y = lst[0]
        match[x] = y
    # make symmetric
    sym: Dict[str, str] = {}
    for x, y in match.items():
        if match.get(y) == x:
            sym[x] = y
    return sym if sym else None

