"""Official PQ gate numbers from the organizer PDF (feet, top-left origin)."""
from __future__ import annotations

# Gate, X_ft, Y_ft. Rotation is unused for identity; Y is down the sheet.
_PDF_FT: dict[int, tuple[float, float]] = {
    1: (12.0, 71.0),
    2: (16.0, 39.0),
    3: (40.0, 17.0),
    4: (72.0, 40.0),
    5: (66.0, 70.0),
    6: (41.0, 86.0),
    7: (70.5, 106.6),
    8: (60.0, 129.0),
    9: (39.7, 147.7),
    10: (13.0, 110.0),
}
_SHEET_Y_FT = 165.0
_FT = 0.3048


def official_xy_m(official_n: int) -> tuple[float, float]:
    x_ft, y_ft = _PDF_FT[int(official_n)]
    return x_ft * _FT, (_SHEET_Y_FT - y_ft) * _FT


def official_number_from_xy(x: float, y: float) -> int:
    """Nearest official gate number to a world XY (metres). G9 top/bottom share XY."""
    best_n = 1
    best_d = float("inf")
    for n, (x_ft, y_ft) in _PDF_FT.items():
        ox, oy = x_ft * _FT, (_SHEET_Y_FT - y_ft) * _FT
        d = (x - ox) ** 2 + (y - oy) ** 2
        if d < best_d:
            best_d = d
            best_n = n
    return best_n


def isaac_index_for_official(gate_xy, official_n: int) -> int:
    """``gate_xy`` is (N, 2+) metres. G9 picks the first of the stacked pair."""
    tx, ty = official_xy_m(official_n)
    best_i = 0
    best_d = float("inf")
    n = len(gate_xy)
    for i in range(n):
        dx = float(gate_xy[i][0]) - tx
        dy = float(gate_xy[i][1]) - ty
        d = dx * dx + dy * dy
        if d < best_d:
            best_d = d
            best_i = i
    return best_i
