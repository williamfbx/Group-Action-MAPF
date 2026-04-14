import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _build_gradient(phi, maze):
	h, w = phi.shape
	grad = np.zeros((h, w, 2), dtype=float)

	for r in range(h):
		for c in range(w):
			if maze[r, c] == 1:
				continue

			left_free = c - 1 >= 0 and maze[r, c - 1] == 0
			right_free = c + 1 < w and maze[r, c + 1] == 0
			up_free = r - 1 >= 0 and maze[r - 1, c] == 0
			down_free = r + 1 < h and maze[r + 1, c] == 0

			if right_free and left_free:
				dphi_dc = (phi[r, c + 1] - phi[r, c - 1]) / 2.0
			elif right_free:
				dphi_dc = phi[r, c + 1] - phi[r, c]
			elif left_free:
				dphi_dc = phi[r, c] - phi[r, c - 1]
			else:
				dphi_dc = 0.0

			if down_free and up_free:
				dphi_dr = (phi[r + 1, c] - phi[r - 1, c]) / 2.0
			elif down_free:
				dphi_dr = phi[r + 1, c] - phi[r, c]
			elif up_free:
				dphi_dr = phi[r, c] - phi[r - 1, c]
			else:
				dphi_dr = 0.0

			# Keep swapped sign convention: plot -grad(phi).
			grad[r, c, 0] = -dphi_dc
			grad[r, c, 1] = -dphi_dr

	return grad


def plot_solution(maze, starts, goals, phi, output_path):
	"""Plot one combined multi-agent figure: map, potential, and gradient."""
	maze_np = np.array(maze, dtype=int)
	phi_np = np.array(phi, dtype=float)
	starts = list(starts)
	goals = list(goals)

	if maze_np.shape != phi_np.shape:
		raise ValueError("maze and phi must have the same shape")

	grad_phi = _build_gradient(phi_np, maze_np)

	fig, axs = plt.subplots(1, 3, figsize=(15, 5))
	cmap_agents = plt.cm.get_cmap("tab10", max(1, len(starts)))
	marker_size = 28

	# Panel 1: original map
	axs[0].imshow(maze_np, cmap="gray_r")
	axs[0].set_title("Original Map")
	axs[0].axis("off")

	# Panel 2: potential field
	im_phi = axs[1].imshow(phi_np, cmap="viridis")
	axs[1].set_title("Potential Field")
	axs[1].axis("off")
	fig.colorbar(im_phi, ax=axs[1], label="phi", fraction=0.046, pad=0.04)

	# Panel 3: gradient field
	axs[2].imshow(phi_np, cmap="viridis")
	axs[2].set_title("Gradient Field")
	axs[2].axis("off")

	y, x = np.mgrid[0:phi_np.shape[0], 0:phi_np.shape[1]]
	grad_mag = np.linalg.norm(grad_phi, axis=2)

	# Normalize to unit direction, then scale by magnitude.
	unit_x = np.zeros_like(grad_phi[:, :, 0])
	unit_y = np.zeros_like(grad_phi[:, :, 1])
	np.divide(grad_phi[:, :, 0], grad_mag, out=unit_x, where=grad_mag > 0)
	np.divide(grad_phi[:, :, 1], grad_mag, out=unit_y, where=grad_mag > 0)
	max_mag = grad_mag.max() if grad_mag.max() > 0 else 1.0
	grad_x = unit_x * (grad_mag / max_mag)
	grad_y = unit_y * (grad_mag / max_mag)

	# Draw one vector per free cell.
	grad_mask = maze_np == 0
	axs[2].quiver(
		x[grad_mask],
		y[grad_mask],
		grad_x[grad_mask],
		grad_y[grad_mask],
		color="w",
		alpha=0.9,
		angles="xy",
		scale_units="xy",
		scale=1.25,
		width=0.003,
		headwidth=4,
		headlength=5,
		headaxislength=4,
		pivot="tail",
		zorder=3,
	)

	legend_handles = []
	for i, (start, goal) in enumerate(zip(starts, goals)):
		color = cmap_agents(i)
		for ax in axs:
			ax.scatter(start[1], start[0], color=color, marker="o", s=marker_size, zorder=4)
			ax.scatter(goal[1], goal[0], color=color, marker="x", s=marker_size + 6, zorder=4)

		legend_handles.append(
			Line2D(
				[0],
				[0],
				color=color,
				marker="o",
				linestyle="None",
				markersize=6,
				label="A{} start".format(i),
			)
		)
		legend_handles.append(
			Line2D(
				[0],
				[0],
				color=color,
				marker="x",
				linestyle="None",
				markersize=6,
				label="A{} goal".format(i),
			)
		)

	# fig.legend(
	# 	handles=legend_handles,
	# 	loc="lower center",
	# 	ncol=min(6, max(1, len(legend_handles))),
	# 	frameon=True,
	# 	bbox_to_anchor=(0.5, -0.02),
	# )
	plt.tight_layout(rect=(0, 0.06, 1, 1))
	plt.savefig(output_path, dpi=220, bbox_inches="tight")
	plt.close(fig)


__all__ = ["plot_solution"]
