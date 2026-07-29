#!/usr/bin/env python3
"""Verify and evaluate the phase-safe recursion starting directly in C_7^5."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from decimal import Decimal, getcontext
from pathlib import Path


Q = 7
PAIRS = (
    ((1, 3, 4, 4, 6), (2, 3, 5, 4, 6)),
    ((3, 4, 0, 3, 5), (2, 4, 6, 3, 5)),
    ((5, 3, 1, 3, 4), (5, 3, 2, 3, 5)),
    ((4, 4, 6, 1, 6), (5, 4, 6, 0, 6)),
    ((6, 0, 6, 4, 5), (6, 1, 6, 5, 5)),
    ((0, 3, 5, 6, 5), (6, 3, 5, 0, 5)),
    ((6, 4, 3, 4, 0), (6, 4, 2, 4, 6)),
    ((6, 4, 5, 3, 2), (6, 5, 5, 3, 1)),
)
J0 = frozenset((0, 5, 6))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--r367",
        type=Path,
        default=root / "inputs/R367.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "phase_recursion_from_C7_5.json",
    )
    parser.add_argument("--balanced-levels", type=int, default=8)
    parser.add_argument("--exact-dp-blocks", type=int, default=64)
    parser.add_argument("--float-scan-blocks", type=int, default=4096)
    return parser.parse_args()


def read_points(path: Path) -> list[tuple[int, ...]]:
    points = [
        tuple(map(int, line.split()))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(len(point) != 5 for point in points):
        raise ValueError("R367 contains a point of the wrong dimension")
    return points


def closed_adjacent(
    left: tuple[int, ...], right: tuple[int, ...]
) -> bool:
    return all(
        (x - y) % Q in (0, 1, Q - 1)
        for x, y in zip(left, right)
    )


def verify_independent(points: list[tuple[int, ...]]) -> None:
    if len(points) != len(set(points)):
        raise AssertionError("duplicate point")
    for index, left in enumerate(points):
        for right in points[index + 1 :]:
            if closed_adjacent(left, right):
                raise AssertionError("conflicting pair")


def transform_t(point: tuple[int, ...]) -> tuple[int, ...]:
    return (
        (2 - point[1]) % Q,
        point[3],
        point[0],
        (2 - point[2]) % Q,
        point[4],
    )


def decimal_rate(cardinality: int, dimension: int) -> str:
    getcontext().prec = 100
    return format(
        (Decimal(cardinality).ln() / Decimal(dimension)).exp(),
        ".70f",
    )


def main() -> None:
    args = parse_args()
    if args.balanced_levels < 1:
        raise ValueError("--balanced-levels must be positive")
    if args.exact_dp_blocks < 40:
        raise ValueError("--exact-dp-blocks must be at least 40")

    r367 = read_points(args.r367)
    if len(r367) != 367:
        raise AssertionError("R367 has the wrong size")
    verify_independent(r367)

    phase_h = {
        PAIRS[index][0] if index in J0 else PAIRS[index][1]
        for index in range(8)
    }
    phase_v = {
        PAIRS[index][1] if index in J0 else PAIRS[index][0]
        for index in range(8)
    }
    if phase_h & phase_v:
        raise AssertionError("the five-dimensional phases overlap")
    verify_independent(sorted(phase_h))
    verify_independent(sorted(phase_v))

    for center, alternative in PAIRS:
        hits = [
            point
            for point in r367
            if closed_adjacent(alternative, point)
        ]
        if hits != [center]:
            raise AssertionError("a five-dimensional pair is not private")

    companion = [transform_t(point) for point in r367]
    if companion[267] != (2, 4, 6, 3, 5):
        raise AssertionError("unexpected R367 ordering")
    companion[267] = (1, 5, 6, 3, 5)
    verify_independent(companion)
    if set(companion) & (phase_h | phase_v):
        raise AssertionError("the companion meets a private endpoint")

    def category(point: tuple[int, ...]) -> str:
        hit_h = any(closed_adjacent(point, p) for p in phase_h)
        hit_v = any(closed_adjacent(point, p) for p in phase_v)
        if hit_h and hit_v:
            return "HV"
        if hit_h:
            return "H"
        if hit_v:
            return "V"
        return "O"

    profile = Counter(category(point) for point in companion)
    expected_profile = Counter({"O": 321, "H": 26, "V": 20})
    if profile != expected_profile:
        raise AssertionError(f"unexpected five-dimensional profile: {profile}")

    # Equal-size phase squaring.
    balanced: list[dict[str, object]] = []
    a, t, s, o, h, v = 367, 8, 367, 321, 26, 20
    for level in range(args.balanced_levels):
        blocks = 1 << level
        dimension = 5 * blocks
        balanced.append(
            {
                "level": level,
                "five_dimensional_blocks": blocks,
                "dimension": dimension,
                "a": a,
                "t": t,
                "s": s,
                "o": o,
                "h": h,
                "v": v,
                "root_bound": decimal_rate(a, dimension),
            }
        )
        a, t, s, o, h, v = (
            (a - t) ** 2 + 2 * t * s,
            2 * t * o,
            s**2,
            o**2 + (h + v) ** 2,
            h * o + o * v,
            v * o + o * h,
        )

    if balanced[1]["a"] != 134753:
        raise AssertionError("the first lift does not reproduce the paper")
    if balanced[2]["a"] != 18184092097:
        raise AssertionError("the twenty-dimensional count changed")

    # Heterogeneous composition of gadgets occupying i and j base blocks:
    #
    # a_{i+j}=(a_i-t_i)(a_j-t_j)+t_i*s_j+s_i*t_j.
    #
    # The companion and private-pair counts depend only on the total number
    # k of five-dimensional blocks:
    # s_k=367^k, o_k=(367^k+275^k)/2,
    # t_k=2(367^k-275^k)/23.
    n = args.exact_dp_blocks
    s_values = [0] * (n + 1)
    o_values = [0] * (n + 1)
    t_values = [0] * (n + 1)
    a_values = [0] * (n + 1)
    split = [0] * (n + 1)
    for blocks in range(1, n + 1):
        s_values[blocks] = 367**blocks
        d_value = 275**blocks
        o_values[blocks] = (s_values[blocks] + d_value) // 2
        numerator = 2 * (s_values[blocks] - d_value)
        if numerator % 23:
            raise AssertionError("nonintegral private-pair formula")
        t_values[blocks] = numerator // 23

    a_values[1] = 367
    for blocks in range(2, n + 1):
        best = -1
        best_left = 0
        for left in range(1, blocks // 2 + 1):
            right = blocks - left
            candidate = (
                (a_values[left] - t_values[left])
                * (a_values[right] - t_values[right])
                + t_values[left] * s_values[right]
                + s_values[left] * t_values[right]
            )
            if candidate > best:
                best = candidate
                best_left = left
        a_values[blocks] = best
        split[blocks] = best_left

    best_exact_blocks = max(
        range(1, n + 1),
        key=lambda blocks: (
            Decimal(a_values[blocks]).ln() / Decimal(5 * blocks)
        ),
    )
    if best_exact_blocks != 40:
        raise AssertionError(
            "the exact DP optimum through the requested range moved"
        )

    # A fast normalized scan provides evidence about where the optimum lies
    # much farther out.  The explicit k=40 construction does not rely on this
    # floating-point scan.
    scan_n = args.float_scan_blocks
    delta = 275.0 / 367.0
    tau = 2.0 / 23.0
    normalized_t = [0.0] * (scan_n + 1)
    normalized_a = [0.0] * (scan_n + 1)
    normalized_a[1] = 1.0
    normalized_t[1] = 8.0 / 367.0
    scan_split = [0] * (scan_n + 1)
    for blocks in range(2, scan_n + 1):
        normalized_t[blocks] = tau * (1.0 - delta**blocks)
        best = -1.0
        best_left = 0
        for left in range(1, blocks // 2 + 1):
            right = blocks - left
            candidate = (
                (normalized_a[left] - normalized_t[left])
                * (normalized_a[right] - normalized_t[right])
                + normalized_t[left]
                + normalized_t[right]
            )
            if candidate > best:
                best = candidate
                best_left = left
        normalized_a[blocks] = best
        scan_split[blocks] = best_left

    scan_best = max(
        range(1, scan_n + 1),
        key=lambda blocks: (
            math.log(367.0) / 5.0
            + math.log(normalized_a[blocks]) / (5.0 * blocks)
        ),
    )
    if scan_best != 40:
        raise AssertionError("the normalized scan optimum moved")

    # Limit of continuing equal-size phase squaring forever.
    getcontext().prec = 100
    d_decimal = Decimal(275) / Decimal(367)
    tau_decimal = Decimal(2) / Decimal(23)
    normalized = Decimal(1)
    limit_rate = Decimal(0)
    for level in range(20):
        blocks = 1 << level
        limit_rate = (
            (
                Decimal(367).ln()
                + normalized.ln() / Decimal(blocks)
            )
            / Decimal(5)
        ).exp()
        normalized_t_level = tau_decimal * (
            Decimal(1) - d_decimal**blocks
        )
        normalized = (
            normalized - normalized_t_level
        ) ** 2 + 2 * normalized_t_level

    k = 40
    result = {
        "status": "verified",
        "five_dimensional_certificate": {
            "R_size": len(r367),
            "private_pairs": 8,
            "phase_H_size": len(phase_h),
            "phase_V_size": len(phase_v),
            "companion_size": len(companion),
            "phase_profile": dict(sorted(profile.items())),
            "HV_count": profile["HV"],
        },
        "balanced_phase_squaring": balanced,
        "continued_balanced_squaring_limit": format(
            limit_rate, ".70f"
        ),
        "heterogeneous_phase_composition": {
            "exact_DP_checked_through_blocks": n,
            "floating_scan_checked_through_blocks": scan_n,
            "scan_best_blocks": scan_best,
            "explicit_best_certificate": {
                "five_dimensional_blocks": k,
                "dimension": 5 * k,
                "split": [split[k], k - split[k]],
                "a": a_values[k],
                "t": t_values[k],
                "s": s_values[k],
                "o": o_values[k],
                "root_bound": decimal_rate(a_values[k], 5 * k),
            },
            "construction_tree_splits": {
                str(blocks): [split[blocks], blocks - split[blocks]]
                for blocks in (2, 3, 5, 10, 20, 40)
            },
        },
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.set_int_max_str_digits(1_000_000)
    main()
