"""HyperLink's bubble tail joins the bubble, for both speakers (0.72.6.rc3).

Reported from an iPhone: the assistant's reply bubble had a dark
triangle with a light rim at its bottom-left corner, and the user's
bubble looked right. BubbleShape drew the tail for the right side,
mirrored it for the left, and added it with ``addPath``. Mirroring
reverses a path's winding, so the assistant's tail wound against its
bubble, and SwiftUI's non-zero fill cancelled the part where the two
overlapped: a hole the colour of the screen behind it.

This reads the tail's coordinates from the Swift, so it follows any
change to the shape, and checks the geometry that made ``union``
necessary. BubbleShapeTests.swift checks the drawn result on a simulator.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "ios" / "HyperLink" / "Sources" / "Views" / "MessageBubble.swift").read_text(encoding="utf-8")
SHAPE = SOURCE[SOURCE.index("struct BubbleShape"):]
SHAPE = SHAPE[:SHAPE.index("\n}\n")]
RADIUS = float(re.search(r"var radius: CGFloat = (\d+)", SHAPE).group(1))


def _points(edge: float, out: float, bottom: float) -> list[tuple[float, float]]:
    """The tail's move-to, control and end points, in source order."""
    found = []
    for x_sign, x_amount, y_part in re.findall(
        r"CGPoint\(x: edge ([+-]) out \* (\w+), y: bottom(?: ([+-] \w+))?\)", SHAPE
    ):
        amount = RADIUS if x_amount == "radius" else float(x_amount)
        x = edge + (1 if x_sign == "+" else -1) * out * amount
        y = bottom
        if y_part:
            sign, value = y_part.split()
            y += (1 if sign == "+" else -1) * (RADIUS if value == "radius" else float(value))
        found.append((x, y))
    return found


def _tail(edge: float, out: float, bottom: float = 60.0, steps: int = 40) -> list[tuple[float, float]]:
    start, c1, p1, c2, p2 = _points(edge, out, bottom)

    def quad(a, c, b):
        return [((1 - t) ** 2 * a[0] + 2 * (1 - t) * t * c[0] + t * t * b[0],
                 (1 - t) ** 2 * a[1] + 2 * (1 - t) * t * c[1] + t * t * b[1])
                for t in (i / steps for i in range(1, steps + 1))]

    return [start, *quad(start, c1, p1), *quad(p1, c2, p2)]


def _signed_area(poly: list[tuple[float, float]]) -> float:
    return sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
               for i in range(len(poly))) / 2


def _inside(point: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    x, y = point
    hit = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            hit = not hit
    return hit


def test_the_coordinates_were_read():
    assert len(_points(0, 1, 60)) == 5


def test_the_mirrored_tails_wind_opposite_ways():
    """Why ``addPath`` could never work for both sides at once."""
    user = _signed_area(_tail(200, 1))
    assistant = _signed_area(_tail(0, -1))
    assert user and assistant and (user > 0) != (assistant > 0)


def test_each_tail_overlaps_its_bubble():
    """The overlap is where the hole was: it is real, not a sliver."""
    overlap = [(x + 0.5, y + 0.5) for x in range(0, 20) for y in range(43, 60)
               if _inside((x + 0.5, y + 0.5), _tail(0, -1))]
    assert len(overlap) > 50


def test_the_tail_is_joined_with_a_union():
    assert "path.union(tail)" in SHAPE
    assert "addPath(tail)" not in SHAPE


def test_both_speakers_still_get_the_same_shape():
    user = _tail(200, 1)
    assistant = _tail(0, -1)
    mirrored = [(200 - x, y) for x, y in assistant]
    assert all(abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9 for a, b in zip(user, mirrored, strict=True))
