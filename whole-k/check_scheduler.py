"""CPU check: whole-K scheduling visits every valid tile exactly once."""
from math import ceil

cases = 0
for grid in (1, 2, 3, 16, 48, 96, 192):
    for total in (1, 7, 16, 31, 32, 63, 64, 95, 96, 127, 128, 255, 256, 416, 1498):
        for k in (16, 32, 48, 96):
            part2, part1 = total, 0
            if total > grid:
                part2 = total % grid
                if part2 * 3 <= grid:
                    part2 += grid
                part1 = (total - part2) // grid
            iters = ceil(ceil(k * part2 / grid) / k) * k
            seen = []
            for cta in range(grid):
                seen += [(i * grid + cta, 0, k, 1) for i in range(part1)]
                start, stop = iters * cta, iters * (cta + 1)
                col, row = start // k, start % k
                while col < part2:
                    amount = min(stop - (k * col + row), k - row)
                    if amount <= 0:
                        break
                    first = iters * ceil(k * col / iters)
                    count = 1
                    if first <= k * (col + 1):
                        offset = first - k * col
                        count = ceil((k - offset) / iters) + (offset > 0)
                    seen.append((total - part2 + col, row, amount, count))
                    col += 1
                    row = 0
            assert sorted(x[0] for x in seen) == list(range(total)), (grid, total, k)
            assert all(row == 0 and amount == k and count == 1 for _, row, amount, count in seen)
            cases += 1
print(f"PASS: {cases} scheduling cases cover every tile once with full K and slice_count=1")
