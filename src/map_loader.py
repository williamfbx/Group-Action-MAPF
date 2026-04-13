import argparse
import os
import random
from typing import List, Tuple


Grid = List[List[bool]]
Coord = Tuple[int, int]


def parse_map_file(map_path: str) -> Grid:
    """Parse a MovingAI .map file into an obstacle grid."""
    with open(map_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    if len(lines) < 5:
        raise ValueError("Invalid .map file: too few lines")

    height = None
    width = None
    map_start = None

    for i, line in enumerate(lines):
        lower = line.strip().lower()
        if lower.startswith("height"):
            height = int(lower.split()[1])
        elif lower.startswith("width"):
            width = int(lower.split()[1])
        elif lower == "map":
            map_start = i + 1
            break

    if height is None or width is None or map_start is None:
        raise ValueError("Invalid .map file: missing height/width/map header")

    raw_grid = lines[map_start: map_start + height]
    if len(raw_grid) != height:
        raise ValueError("Invalid .map file: map body size does not match height")

    traversable = {".", "G", "S"}
    grid: Grid = []

    for row in raw_grid:
        if len(row) != width:
            raise ValueError("Invalid .map file: row width does not match header")

        grid.append([ch not in traversable for ch in row])

    return grid


def sample_agent_positions(grid: Grid, num_agents: int, seed: int = None) -> Tuple[List[Coord], List[Coord]]:
    """Sample random non-overlapping start/goal cells."""
    if num_agents <= 0:
        raise ValueError("num_agents must be > 0")

    free_cells: List[Coord] = [
        (r, c)
        for r in range(len(grid))
        for c in range(len(grid[0]))
        if not grid[r][c]
    ]

    needed = 2 * num_agents
    if len(free_cells) < needed:
        raise ValueError(
            "Not enough free cells for {} agents (need {}, have {}).".format(
                num_agents, needed, len(free_cells)
            )
        )

    rng = random.Random(seed)
    selected = rng.sample(free_cells, needed)
    starts = selected[:num_agents]
    goals = selected[num_agents:]
    return starts, goals


def write_instance_txt(output_path: str, grid: Grid, starts: List[Coord], goals: List[Coord]) -> None:
    rows = len(grid)
    cols = len(grid[0])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("{} {}\n".format(rows, cols))

        for r in range(rows):
            row_tokens = ["@" if grid[r][c] else "." for c in range(cols)]
            f.write(" ".join(row_tokens) + "\n")

        f.write("{}\n".format(len(starts)))

        for (sx, sy), (gx, gy) in zip(starts, goals):
            f.write("{} {} {} {}\n".format(sx, sy, gx, gy))


def default_output_file(src_dir: str, map_path: str, num_agents: int) -> str:
    map_name = os.path.splitext(os.path.basename(map_path))[0]
    out_dir = os.path.join(src_dir, "benchmarks", map_name)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "agents_{}".format(num_agents))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a .map file into TAPF instance .txt format with random agents."
    )
    parser.add_argument("map_path", help="Path to .map file")
    parser.add_argument("num_agents", type=int, help="Number of random agents to initialize")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible start/goal generation",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output .txt path. If omitted, writes to "
            "src/benchmarks/{map_name}/agents_{num_agents}"
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    map_path = args.map_path
    if not os.path.isabs(map_path):
        map_path = os.path.abspath(os.path.join(project_root, map_path))

    if not os.path.exists(map_path):
        raise FileNotFoundError("Map file not found: {}".format(map_path))

    grid = parse_map_file(map_path)
    starts, goals = sample_agent_positions(grid, args.num_agents, args.seed)

    output_path = args.output
    if output_path is None:
        output_path = default_output_file(script_dir, map_path, args.num_agents)
    elif not os.path.isabs(output_path):
        output_path = os.path.abspath(os.path.join(project_root, output_path))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_instance_txt(output_path, grid, starts, goals)

    print("Wrote instance to: {}".format(output_path))


if __name__ == "__main__":
    main()
