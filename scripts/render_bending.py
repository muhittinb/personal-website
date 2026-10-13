"""Render the supplied VTU results into a website video.

Run with a Python environment containing meshio, numpy, matplotlib and Pillow:
    /opt/anaconda3/bin/python scripts/render_bending.py
Requires ffmpeg on PATH. No solver or synthetic crack path is used.
"""

import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bending-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon, Rectangle
from matplotlib.tri import Triangulation
import meshio
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "3p_website"
DESTINATION = ROOT / "fig"
SCALE = 15
FPS = 12
WIDTH, HEIGHT = 1280, 640


def series(name):
    entries = ET.parse(SOURCE / f"{name}.pvd").findall(".//DataSet")
    if not entries:
        raise ValueError(f"No results in {name}.pvd")
    return [(float(item.attrib["timestep"]), SOURCE / item.attrib["file"]) for item in entries]


def main():
    damage_files, displacement_files = series("damage"), series("displacement")
    if len(damage_files) != len(displacement_files):
        raise ValueError("Damage and displacement frame counts differ")
    first = meshio.read(damage_files[0][1])
    points, cells = first.points, first.cells_dict["triangle"]
    xmin, ymin = points[:, :2].min(axis=0)
    xmax, ymax = points[:, :2].max(axis=0)
    middle = (xmin + xmax) / 2
    support_nodes = [np.argmin(np.linalg.norm(points[:, :2] - position, axis=1))
                     for position in ((xmin, ymin), (xmax, ymin))]
    load_node = np.argmin(np.linalg.norm(points[:, :2] - (middle, ymax), axis=1))
    # The central notch tip is at half the specimen height in this supplied mesh.
    zoom_mask = np.any(np.abs(points[cells, 0] - middle) > 30, axis=1)
    cmap = LinearSegmentedColormap.from_list("damage", ["#e5edf4", "#9bb9d1", "#f3ab64", "#bf3327", "#611926"])
    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100, facecolor="#fbfcfe")
    beam_ax = fig.add_axes([0.04, 0.48, 0.92, 0.40])
    zoom_ax = fig.add_axes([0.30, 0.09, 0.40, 0.31])
    color_ax = fig.add_axes([0.79, 0.19, 0.13, 0.02])
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
                       cax=color_ax, orientation="horizontal", ticks=[0, 1])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, colors="#40556a", labelsize=11)
    color_ax.set_title("d", loc="left", fontsize=14, color="#243f58", pad=12)
    fig.text(0.955, 0.94, f"u × {SCALE}", fontsize=13, color="#60758a", ha="right")
    progress = Rectangle((0.045, 0.045), 0, 0.006, transform=fig.transFigure, color="#537696", linewidth=0)
    fig.add_artist(Rectangle((0.045, 0.045), 0.91, 0.006, transform=fig.transFigure, color="#e5edf4", linewidth=0))
    fig.add_artist(progress)
    process = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an",
        "-c:v", "libx264", "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(DESTINATION / "three-point-bending.mp4")
    ], stdin=subprocess.PIPE)
    try:
        for index, ((damage_time, damage_file), (displacement_time, displacement_file)) in enumerate(zip(damage_files, displacement_files)):
            if not np.isclose(damage_time, displacement_time):
                raise ValueError(f"Unmatched frame {index}")
            damage_mesh, displacement_mesh = meshio.read(damage_file), meshio.read(displacement_file)
            for mesh in (damage_mesh, displacement_mesh):
                if not np.allclose(mesh.points, points) or not np.array_equal(mesh.cells_dict["triangle"], cells):
                    raise ValueError(f"Mesh ordering changed at frame {index}")
            damage = damage_mesh.point_data["damage"]
            displacement = displacement_mesh.point_data["displacement"]
            if not np.isfinite(damage).all() or not np.isfinite(displacement).all():
                raise ValueError(f"Non-finite values in frame {index}")
            if damage.min() < -1e-5 or damage.max() > 1 + 1e-5:
                raise ValueError(f"Damage outside 0–1 at frame {index}")
            deformed = points[:, :2] + SCALE * displacement[:, :2]
            triangulation = Triangulation(deformed[:, 0], deformed[:, 1], cells)
            for ax in (beam_ax, zoom_ax):
                ax.clear()
                ax.set_aspect("equal")
                ax.set_axis_off()
            beam_ax.tripcolor(triangulation, damage, shading="gouraud", cmap=cmap, vmin=0, vmax=1)
            beam_ax.set_xlim(xmin - 20, xmax + 35)
            beam_ax.set_ylim(ymin - 30, ymax + 45)
            # Symbols mark the constrained bottom corners and imposed top displacement.
            for support_index, node in enumerate(support_nodes):
                x, y = deformed[node]
                beam_ax.add_patch(Polygon([(x, y - 2), (x - 7, y - 14), (x + 7, y - 14)],
                                          facecolor="#fbfcfe", edgecolor="#537696", linewidth=1.6))
                if support_index == 0:
                    # Pinned support: the triangle sits directly on hatched ground.
                    ground_y = y - 14
                    beam_ax.plot([x - 12, x + 12], [ground_y, ground_y], color="#537696", linewidth=1.6)
                    for offset in range(-10, 13, 4):
                        beam_ax.plot([x + offset - 3, x + offset],
                                     [ground_y - 4, ground_y], color="#537696", linewidth=1)
                else:
                    beam_ax.plot([x - 12, x + 12], [y - 18, y - 18], color="#537696", linewidth=1.6)
            x, y = deformed[load_node]
            beam_ax.annotate("", xy=(x, y + 2), xytext=(x, y + 29),
                             arrowprops=dict(arrowstyle="-|>", color="#243f58", lw=2, mutation_scale=15))
            beam_ax.add_patch(Rectangle((middle - 25, 43), 65, 61, fill=False,
                                        edgecolor="#537696", linestyle=(0, (3, 3)), linewidth=1))
            zoom_tri = Triangulation(deformed[:, 0], deformed[:, 1], cells, mask=zoom_mask)
            zoom_ax.tripcolor(zoom_tri, damage, shading="gouraud", cmap=cmap, vmin=0, vmax=1)
            zoom_ax.set_xlim(middle - 14, middle + 27)
            zoom_ax.set_ylim(43, 104)
            zoom_ax.set_frame_on(True)
            for spine in zoom_ax.spines.values():
                spine.set_edgecolor("#d5e1eb")
            progress.set_width(0.91 * index / (len(damage_files) - 1))
            fig.canvas.draw()
            pixels = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            if index == len(damage_files) - 1:
                Image.fromarray(pixels).save(DESTINATION / "three-point-bending.webp", quality=90)
            repeat = FPS + 1 if index in (0, len(damage_files) - 1) else 1
            for _ in range(repeat):
                process.stdin.write(pixels.tobytes())
            if index % 20 == 0 or index == len(damage_files) - 1:
                print(f"Rendered {index + 1}/{len(damage_files)} frames", flush=True)
    finally:
        process.stdin.close()
        result = process.wait()
        plt.close(fig)
    if result:
        raise RuntimeError(f"ffmpeg exited with {result}")
    print(f"Created {DESTINATION / 'three-point-bending.mp4'}", flush=True)


if __name__ == "__main__":
    main()
