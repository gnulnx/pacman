#!/usr/bin/env python3
"""
pacman_macro_mapgen.py
Author: ChatGPT

Generates Pac-Man–style mazes using a macro-tile ("dojo") system
and renders multiple mazes into a single image.

Usage:
  python pacman_macro_mapgen.py -n 16 --cols 4 --rows 4 --out mazes.png --seed 123
"""

import argparse
import colorsys
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw

# ---------------------- Global map sizing ----------------------
# Target classic-ish size: 28 x 36 (WxH)
MAP_W, MAP_H = 28, 36

# We compose the LEFT HALF as a grid of macro-tiles,
# then mirror horizontally to form the right half.
# Each macro-tile has size TILE_H x TILE_W (rows x cols) on the grid.
# Choose sizes that exactly tile MAP_H and MAP_W//2.
TILE_H, TILE_W = 6, 7  # 6*6 = 36 rows, 7*2 = 14 cols (half of 28)
TILES_H = MAP_H // TILE_H  # 6
TILES_W_HALF = (MAP_W // 2) // TILE_W  # 2

# Rendering pixel size per cell
PX = 8


# ---------------------- Helpers ----------------------
def hsl_to_rgb(h: float, s: float, l: float) -> Tuple[int, int, int]:
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (int(r * 255), int(g * 255), int(b * 255))


def random_wall_palette(seed=None) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    rnd = random.Random(seed)
    hue = rnd.random()
    wall_fill = hsl_to_rgb(hue, 1.0, 0.60)
    wall_stroke = hsl_to_rgb((hue + 0.55) % 1.0, 0.6, 0.65)
    return wall_fill, wall_stroke


# ---------------------- Macro-tile system ----------------------
# Tile characters:
# ' ' = path / floor, '.' = pellet placeholder (converted later), '|' = wall
# We will ensure all floor inside tiles touch the tile edges only at specific connector slots.
#
# Each macro-tile declares which edge slots are "open" (floor reaches boundary).
# For simplicity, we place potential connectors at fixed offsets:
#  - Left/Right edges: y in {2, 3} (0-indexed within tile height)
#  - Top/Bottom edges: x in {3} (middle column inside 7-wide tile)
#
LEFT_SLOTS = (2, 3)
RIGHT_SLOTS = (2, 3)
TOP_SLOTS = (3,)
BOT_SLOTS = (3,)


@dataclass(frozen=True)
class MacroTile:
    name: str
    # 2D pattern (rows of length TILE_W)
    grid: Tuple[str, ...]
    # Open connectors on edges: indices relative to that edge
    open_left: Tuple[int, ...]
    open_right: Tuple[int, ...]
    open_top: Tuple[int, ...]
    open_bot: Tuple[int, ...]

    def rotated(self, k: int = 0) -> "MacroTile":
        """Return a tile rotated 90*k degrees (k in {0,1,2,3})."""
        g = [list(row) for row in self.grid]
        for _ in range(k % 4):
            # rotate 90 deg clockwise
            g = [list(row) for row in zip(*g[::-1])]
        # ensure dimensions stay TILE_H x TILE_W
        if len(g) != TILE_H or len(g[0]) != TILE_W:
            # we don't rotate non-square here; only return same orientation
            return self

        # recompute connectors by rotating coordinate frame
        def rot_edge(open_set, edge):
            # edge: 'L','R','T','B'
            # we know allowed slot coords on each edge:
            # left/right: y indices; top/bot: x indices
            out = []
            for v in open_set:
                if edge == "L":
                    # slot at (x=0, y=v)
                    x, y = 0, v
                elif edge == "R":
                    x, y = TILE_W - 1, v
                elif edge == "T":
                    x, y = v, 0
                else:
                    x, y = v, TILE_H - 1
                # rotate 90 deg clockwise around tile
                xr, yr = y, TILE_W - 1 - x
                out.append((xr, yr))
            # Map rotated coordinates back to slots on new edges
            # Determine new edge per rotated location
            left, right, top, bot = [], [], [], []
            for xr, yr in out:
                if xr == 0:
                    top.append(yr)  # since after rotation, x-axis swapped, careful, but we only use existing sets
                elif xr == TILE_W - 1:
                    bot.append(yr)
                elif yr == 0:
                    left.append(xr)
                elif yr == TILE_H - 1:
                    right.append(xr)
            # We cannot robustly generalize across non-square; skip rotation usage.
            return ()

        # For non-square tiles, avoid rotation to keep connectors consistent.
        return MacroTile(
            self.name, tuple("".join(row) for row in g), self.open_left, self.open_right, self.open_top, self.open_bot
        )


def pad_room(pattern: List[str]) -> Tuple[str, ...]:
    """Ensure pattern is TILE_H x TILE_W."""
    out = []
    for r in pattern[:TILE_H]:
        r = (r + " " * TILE_W)[:TILE_W]
        out.append(r)
    while len(out) < TILE_H:
        out.append(" " * TILE_W)
    return tuple(out)


# Define a small library of tiles (left-half primitives).
# Ensure internal corridors connect to edges only at declared slots.
TILES: List[MacroTile] = []

# Straight vertical corridor through center
TILES.append(
    MacroTile(
        "I_v",
        pad_room(
            [
                "   |   ",
                "   |   ",
                "   |   ",
                "   |   ",
                "   |   ",
                "   |   ",
            ]
        ),
        open_left=(),
        open_right=(),
        open_top=TOP_SLOTS,
        open_bot=BOT_SLOTS,
    )
)

# Straight horizontal corridor mid rows
TILES.append(
    MacroTile(
        "I_h",
        pad_room(
            [
                "       ",
                " ||||| ",
                "       ",
                " ||||| ",
                "       ",
                "       ",
            ]
        ),
        open_left=LEFT_SLOTS,
        open_right=RIGHT_SLOTS,
        open_top=(),
        open_bot=(),
    )
)

# Corner: connects top -> right (turn)
TILES.append(
    MacroTile(
        "L_tr",
        pad_room(
            [
                "   |   ",
                "   |   ",
                "   |   ",
                " ||    ",
                "       ",
                "       ",
            ]
        ),
        open_left=(),
        open_right=(2,),  # right slot at y=2 (approx mid)
        open_top=(3,),
        open_bot=(),
    )
)

# Corner: connects left -> bot
TILES.append(
    MacroTile(
        "L_lb",
        pad_room(
            [
                "       ",
                "       ",
                "    || ",
                "   |   ",
                "   |   ",
                "   |   ",
            ]
        ),
        open_left=(3,),
        open_right=(),
        open_top=(),
        open_bot=(3,),
    )
)

# T junction: left + right + top
TILES.append(
    MacroTile(
        "T_top",
        pad_room(
            [
                "   |   ",
                "   |   ",
                "|||||||",
                "       ",
                "       ",
                "       ",
            ]
        ),
        open_left=LEFT_SLOTS,
        open_right=RIGHT_SLOTS,
        open_top=(3,),
        open_bot=(),
    )
)

# Cross (+): all directions
TILES.append(
    MacroTile(
        "Cross",
        pad_room(
            [
                "   |   ",
                "   |   ",
                "|||||||",
                "   |   ",
                "   |   ",
                "       ",
            ]
        ),
        open_left=LEFT_SLOTS,
        open_right=RIGHT_SLOTS,
        open_top=(3,),
        open_bot=(3,),
    )
)

# Empty / room (few internal walls)
TILES.append(
    MacroTile(
        "Room",
        pad_room(
            [
                "|||||||",
                "|     |",
                "|     |",
                "|     |",
                "|     |",
                "|||||||",
            ]
        ),
        open_left=(),
        open_right=(),
        open_top=(),
        open_bot=(),
    )
)


# ---------------------- Composition ----------------------
def compose_left_half(seed: Optional[int] = None) -> List[List[str]]:
    rnd = random.Random(seed)
    half = [[" " for _ in range(MAP_W // 2)] for _ in range(MAP_H)]

    # choose a tile for each macro cell with simple adjacency constraints:
    # - horizontally: right openings must match left openings of neighbor
    # - vertically: bottom openings must match top openings of cell below
    # to keep it simple, we randomly resample until constraints satisfied (grid is small).
    grid_tiles: List[List[MacroTile]] = [[None for _ in range(TILES_W_HALF)] for _ in range(TILES_H)]

    def compatible_h(a: MacroTile, b: MacroTile) -> bool:
        if a is None or b is None:
            return True
        return bool(a.open_right) == bool(b.open_left)

    def compatible_v(a: MacroTile, b: MacroTile) -> bool:
        if a is None or b is None:
            return True
        return bool(a.open_bot) == bool(b.open_top)

    for r in range(TILES_H):
        for c in range(TILES_W_HALF):
            tries = 0
            while True:
                tile = rnd.choice(TILES)
                left_ok = compatible_h(grid_tiles[r][c - 1], tile) if c > 0 else True
                up_ok = compatible_v(grid_tiles[r - 1][c], tile) if r > 0 else True
                # discourage Room on borders (keeps perimeter open for walls later)
                if (r in (0, TILES_H - 1) or c == 0) and tile.name == "Room":
                    cond_room = rnd.random() < 0.2
                else:
                    cond_room = True
                if left_ok and up_ok and cond_room:
                    grid_tiles[r][c] = tile
                    break
                tries += 1
                if tries > 50:
                    grid_tiles[r][c] = rnd.choice(TILES)
                    break

    # paint tiles into half-grid
    for r in range(TILES_H):
        for c in range(TILES_W_HALF):
            t = grid_tiles[r][c]
            base_y = r * TILE_H
            base_x = c * TILE_W
            for dy in range(TILE_H):
                row = t.grid[dy]
                for dx in range(TILE_W):
                    ch = row[dx]
                    if ch != " ":
                        half[base_y + dy][base_x + dx] = ch

            # carve connectors into edges if declared
            # left
            for yslot in t.open_left:
                y = base_y + yslot
                x = base_x
                half[y][x] = " "
            # right
            for yslot in t.open_right:
                y = base_y + yslot
                x = base_x + TILE_W - 1
                half[y][x] = " "
            # top
            for xslot in t.open_top:
                x = base_x + xslot
                y = base_y
                half[y][x] = " "
            # bottom
            for xslot in t.open_bot:
                x = base_x + xslot
                y = base_y + TILE_H - 1
                half[y][x] = " "

    return half


def mirror_and_finalize(half: List[List[str]]) -> List[List[str]]:
    w_half = MAP_W // 2
    full = [[" " for _ in range(MAP_W)] for _ in range(MAP_H)]
    for y in range(MAP_H):
        for x in range(w_half):
            ch = half[y][x]
            full[y][x] = ch
            full[y][MAP_W - 1 - x] = ch  # mirror

    # Outer border walls (keep side tunnels clear at mid rows)
    for x in range(MAP_W):
        full[0][x] = "|"
        full[MAP_H - 1][x] = "|"
    for y in range(MAP_H):
        full[y][0] = "|"
        full[y][MAP_W - 1] = "|"

    # Ghost box in center
    box_w, box_h = 8, 4
    bx0 = MAP_W // 2 - box_w // 2
    by0 = MAP_H // 2 - box_h // 2
    for yy in range(box_h):
        for xx in range(box_w):
            if yy in (0, box_h - 1) or xx in (0, box_w - 1):
                full[by0 + yy][bx0 + xx] = "|"
            else:
                full[by0 + yy][bx0 + xx] = " "
    # ghost door
    full[by0 + box_h - 1][MAP_W // 2] = " "

    # Place pellets on floor cells (sparsify a bit)
    for y in range(1, MAP_H - 1):
        for x in range(1, MAP_W - 1):
            if full[y][x] == " ":
                # avoid placing inside ghost box edges
                if by0 <= y < by0 + box_h and bx0 <= x < bx0 + box_w:
                    continue
                full[y][x] = "." if ((x + y) % 2 == 0) else " "

    # Power pellets near corners (find nearest floor points)
    corners = [(1, 1), (1, MAP_H - 2), (MAP_W - 2, 1), (MAP_W - 2, MAP_H - 2)]
    for cx, cy in corners:
        for r in range(1, 8):
            placed = False
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    x, y = cx + dx, cy + dy
                    if 1 <= x < MAP_W - 1 and 1 <= y < MAP_H - 1:
                        if full[y][x] == ".":
                            full[y][x] = "o"
                            placed = True
                            break
                if placed:
                    break
            if placed:
                break

    # Pac-Man start (left of ghost box)
    full[MAP_H - 5][2] = "P"
    return full


def generate_maze(seed: Optional[int] = None) -> List[List[str]]:
    half = compose_left_half(seed)
    return mirror_and_finalize(half)


# ---------------------- Rendering ----------------------
def draw_one_maze(img: Image.Image, top_left: Tuple[int, int], maze: List[List[str]], wall_fill, wall_stroke):
    draw = ImageDraw.Draw(img)
    x0, y0 = top_left
    # background panel
    draw.rectangle([x0, y0, x0 + MAP_W * PX, y0 + MAP_H * PX], fill=(0, 0, 0))

    pellet_color = (255, 184, 174)

    # draw walls as solid blocks
    for y in range(MAP_H):
        for x in range(MAP_W):
            ch = maze[y][x]
            px = x0 + x * PX
            py = y0 + y * PX
            if ch == "|":
                draw.rectangle([px, py, px + PX - 1, py + PX - 1], fill=wall_fill, outline=wall_stroke, width=1)

    # pellets
    for y in range(MAP_H):
        for x in range(MAP_W):
            ch = maze[y][x]
            if ch in (".", "o"):
                px = x0 + x * PX + PX // 2
                py = y0 + y * PX + PX // 2
                r = 2 if ch == "." else 3
                draw.ellipse([px - r, py - r, px + r, py + r], fill=pellet_color)


def render_grid(n: int, cols: int, rows: int, seed: Optional[int], out_path: str = "mazes.png"):
    rnd = random.Random(seed)
    pad = PX * 2
    w = cols * MAP_W * PX + (cols + 1) * pad
    h = rows * MAP_H * PX + (rows + 1) * pad
    img = Image.new("RGB", (w, h), (30, 30, 30))

    for i in range(n):
        r = i // cols
        c = i % cols
        if r >= rows:
            break
        mz = generate_maze(seed=rnd.randint(0, 10**9))
        wall_fill, wall_stroke = random_wall_palette(rnd.randint(0, 10**9))
        ox = pad + c * (MAP_W * PX + pad)
        oy = pad + r * (MAP_H * PX + pad)
        draw_one_maze(img, (ox, oy), mz, wall_fill, wall_stroke)

    img.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=12, help="number of mazes")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="mazes.png")
    args = ap.parse_args()
    path = render_grid(args.n, args.cols, args.rows, args.seed, args.out)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
