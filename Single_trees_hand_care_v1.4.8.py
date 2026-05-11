import matplotlib
matplotlib.use('Qt5Agg')  # ← must be BEFORE importing pyplot

import os
import glob
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import sys

import numpy as np
import laspy
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector, Button
from matplotlib.path import Path
import matplotlib.colors as mcolors

from PyQt5.QtWidgets import (QApplication, QDialog, QVBoxLayout,
                              QPushButton, QLabel, QFrame, QHBoxLayout, QWidget)
from PyQt5.QtCore import Qt
# remove: import tkinter as tk

# ----------------------------
# Configuration
# ----------------------------
NEIGHBOR_MARGIN = 3.0     # meters around the current tree bbox to show neighbors
SIDE_MARGIN = 0.5
NOISE_ID = 999            # points reclassified as noise
UNCLASSIFIED_ID = 0       # points that are not assigned to any tree


@dataclass
class PendingReassignment:
    """A not-yet-applied reassignment (kept until 'Apply & Stay' or final save)."""
    indices: np.ndarray      # global indices in the LAS/LAZ file
    new_id: int              # target tree ID (or NOISE_ID)
    coords: np.ndarray       # XYZ coordinates of the moved points (for visualization)


class TreeInspector:
    def __init__(self, tree_dir: str, out_dir: Optional[str] = None):
        self.tree_dir = tree_dir
        self.out_dir = out_dir if out_dir is not None else tree_dir
        self.next_new_tree_id: Optional[int] = None
        
        # os.makedirs(self.out_dir, exist_ok=True)

        # Control flags
        self.stop_flag: bool = False
        self.save_requested: bool = False
        self.processing_selection: bool = False  # True while a lasso selection is being processed
        self._queued_actions: List[str] = []     # actions requested while selection is processing

        # Current tree state (per tree id)
        self.current_pc: Optional[np.ndarray] = None          # (N, 3) xyz of current tree (view)
        self.current_indices: Optional[np.ndarray] = None     # (N,) global indices of current tree (view)
        self._original_tree_indices: Optional[np.ndarray] = None

        # Neighbor state (per tree id)
        self.neighbor_pc: np.ndarray = np.empty((0, 3), dtype=float)
        self.neighbor_ids: np.ndarray = np.empty((0,), dtype=int)

        # Undo stack: list of snapshots (pc, indices, reassignments)
        self.history: List[dict] = []
        self.max_history = 20

        # Reassignment buffers
        self.reassignments: List[PendingReassignment] = []          # pending, not applied visually
        self.finalized_reassignments: List[Tuple[np.ndarray, int]] = []  # committed via "Apply & Stay"
        self._highlight_mask = None   # temporary lasso selection

        # Color mapping for neighbor ids (and noise)
        self.id_colors: Dict[int, str] = {}

        # Matplotlib artists
        self.fig = None
        self.axs = None
        self._selectors = []
        self._main_lines = []                     # 3 Line2D objects for the main tree
        self._neighbor_lines: List[Dict[int, object]] = []  # per view dict: id -> Line2D

        # Bounds used for neighbor view filtering (kept fixed per tree)
        self._tree_bounds = None  # (x_min, x_max, y_min, y_max)
        self._qapp = QApplication.instance() or QApplication(sys.argv)

    # ----------------------------
    # Main loop over files/trees
    # ----------------------------
    def run(self):
        laz_files = [
            f for f in sorted(glob.glob(os.path.join(self.tree_dir, "*.laz")))
        ]
        if not laz_files:
            print(f"No .laz files found in: {self.tree_dir}")
            return

        for f_path in laz_files:
            if self.stop_flag:
                break

            print(f"\nProcessing {os.path.basename(f_path)}")

            las = laspy.read(f_path)

            # Copy to a mutable numpy array (we'll write it back at the end)
            tree_ids = np.asarray(las.PredInstance).copy()
            

            # Cache xyz as numpy arrays (faster than repeatedly indexing las.x/las.y/las.z)
            x = np.asarray(las.x)
            y = np.asarray(las.y)
            z = np.asarray(las.z)

            unique_ids = np.unique(tree_ids)
            print("trees", unique_ids)
            unique_ids = unique_ids[(unique_ids > 0) & (unique_ids != NOISE_ID)]  # don't inspect noise as a "tree"
            unique_ids = np.sort(unique_ids)
            self.next_new_tree_id = int(unique_ids.max() + 1) if unique_ids.size else 1

            for tid in unique_ids:
                if self.stop_flag:
                    break

                mask_main = (tree_ids == tid)
                original_tree_indices = np.flatnonzero(mask_main)
                if original_tree_indices.size == 0:
                    continue

                # 1) Main tree arrays
                self._original_tree_indices = original_tree_indices
                self.current_pc = np.column_stack((x[mask_main], y[mask_main], z[mask_main])).astype(float, copy=False)
                self.current_indices = original_tree_indices.copy()

                # Fixed bounds for this tree (used for neighbor filtering in views)
                x_min, x_max = float(self.current_pc[:, 0].min()), float(self.current_pc[:, 0].max())
                y_min, y_max = float(self.current_pc[:, 1].min()), float(self.current_pc[:, 1].max())
                self._tree_bounds = (x_min, x_max, y_min, y_max)

                # 2) Neighbor arrays (points around the tree bbox)
                m = NEIGHBOR_MARGIN
                
                mask_neighbors = (
                    (tree_ids != tid)
                    & (tree_ids != UNCLASSIFIED_ID)
                    & (x >= (x_min - m))
                    & (x <= (x_max + m))
                    & (y >= (y_min - m))
                    & (y <= (y_max + m))
                )

                if np.any(mask_neighbors):
                    self.neighbor_pc = np.column_stack((x[mask_neighbors], y[mask_neighbors], z[mask_neighbors])).astype(float, copy=False)
                    self.neighbor_ids = tree_ids[mask_neighbors].astype(int, copy=False)
                else:
                    self.neighbor_pc = np.empty((0, 3), dtype=float)
                    self.neighbor_ids = np.empty((0,), dtype=int)

                # Reset per-tree interaction buffers
                self.reassignments = []
                self.finalized_reassignments = []
                self.history = []
                self.save_requested = False
                self.processing_selection = False
                self._queued_actions = []

                # Open GUI for this tree id
                self.open_gui(f_path, int(tid))

                print(
                    "AFTER GUI close: len(reassignments) =",
                    len(self.reassignments),
                    "len(finalized_reassignments) =",
                    len(self.finalized_reassignments),
                )

                if self.save_requested:
                    to_apply: List[Tuple[np.ndarray, int]] = []
                    to_apply.extend(self.finalized_reassignments)
                    to_apply.extend((r.indices, int(r.new_id)) for r in self.reassignments)

                    print("run: to_apply has", len(to_apply), "entries")

                    # Apply reassignments
                    for idx_list, new_id in to_apply:
                        tree_ids[idx_list] = new_id

                    # Any points removed from the view but not present in reassignments => mark noise
                    # (This is a safety-net; normally it should be empty.)
                    if self._original_tree_indices is not None and self.current_indices is not None:
                        removed = np.setdiff1d(self._original_tree_indices, self.current_indices, assume_unique=False)

                        if to_apply:
                            reassigned_flat = np.concatenate([idx for idx, _nid in to_apply]).astype(int, copy=False)
                        else:
                            reassigned_flat = np.empty((0,), dtype=int)

                        remaining_removed = np.setdiff1d(removed, reassigned_flat, assume_unique=False)
                        if remaining_removed.size > 0:
                            tree_ids[remaining_removed] = NOISE_ID

            # Save updated LAZ
            las.PredInstance = tree_ids
            base = os.path.splitext(os.path.basename(f_path))[0]
            out_path = os.path.join(self.out_dir, f"{base}_updated.laz")

            las.write(out_path)
            print(f"Saved: {out_path}")

    # ----------------------------
    # GUI building
    # ----------------------------
    def open_gui(self, filename: str, tid: int):
        self._init_id_colors(tid)

        self.fig, self.axs = plt.subplots(1, 3, figsize=(18, 7), facecolor="black")
        self.fig.canvas.manager.set_window_title(f"Tree ID {tid} | Use Scroll to Zoom")

        titles = ["XY Top View", "XZ Front View", "YZ Side View"]
        self._dims = [(0, 1), (0, 2), (1, 2)]  # used by update methods

        # Create empty structures to store artists
        self._selectors = []
        self._main_lines = []
        self._highlight_lines = []

        self._neighbor_lines = [dict(), dict(), dict()]
        self._highlight_mask = np.zeros(len(self.current_pc), dtype=bool)
        for i, (ax, title, dims) in enumerate(zip(self.axs, titles, self._dims)):
            ax.set_facecolor("black")
            ax.grid(True, color="#222222", linestyle="--", linewidth=0.5)
            ax.tick_params(colors="white", labelsize=7)
            ax.set_title(title, color="white", fontsize=10)

            # Neighbor lines (persistent artists)
            self._update_neighbors_for_view(view_index=i, dims=dims, create_if_missing=True)

            # Main tree line
            main_line, = ax.plot(
                self.current_pc[:, dims[0]] if self.current_pc is not None else [],
                self.current_pc[:, dims[1]] if self.current_pc is not None else [],
                ",",
                color="#00FF00",
                alpha=0.7,
                zorder=2,
            )

            highlight_line, = ax.plot([], [], ",",
                                    color="#FFFF00",
                                    alpha=1.0,
                                    zorder=5)
            self._highlight_lines.append(highlight_line)

            

            self._main_lines.append(main_line)

            ax.set_aspect("equal", "datalim")

            # Lasso selector
            try:
                selector = LassoSelector(
                    ax,
                    onselect=lambda verts, d=dims: self.on_select_with_highlight(verts, d),
                    useblit=True
                )

            except TypeError:
                selector = LassoSelector(ax, onselect=self.on_select_with_highlight(dims))
            self._selectors.append(selector)

        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self._setup_buttons()

        # Safety: if the user closes the window using the window manager X button
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        plt.show()

    def _setup_buttons(self):
        plt.subplots_adjust(bottom=0.2)

        ax_undo = plt.axes([0.05, 0.05, 0.08, 0.05])
        ax_apply = plt.axes([0.15, 0.05, 0.15, 0.05])
        ax_erase = plt.axes([0.32, 0.05, 0.1, 0.05])
        ax_save = plt.axes([0.44, 0.05, 0.15, 0.05])
        ax_skip = plt.axes([0.61, 0.05, 0.1, 0.05])
        ax_stop = plt.axes([0.75, 0.05, 0.1, 0.05])

        self.btn_undo = Button(ax_undo, "Undo")
        self.btn_apply = Button(ax_apply, "Apply & Stay", color="#3498db", hovercolor="#2980b9")
        self.btn_erase = Button(ax_erase, "Erase Tree")
        self.btn_save = Button(ax_save, "Save & Next", color="#2ecc71")
        self.btn_skip = Button(ax_skip, "Skip")
        self.btn_stop = Button(ax_stop, "Exit")

        self.btn_apply.on_clicked(self.apply_changes)
        self.btn_undo.on_clicked(self.undo)
        self.btn_erase.on_clicked(self.erase_tree)
        self.btn_save.on_clicked(self.save_exit)
        self.btn_skip.on_clicked(self.skip_exit)
        self.btn_stop.on_clicked(self.stop_all)

    def _update_highlight_display(self):
        if self._highlight_mask is None:
            return

        for i, dims in enumerate(self._dims):
            pts = self.current_pc[self._highlight_mask]
            self._highlight_lines[i].set_data(
                pts[:, dims[0]],
                pts[:, dims[1]]
            )

        self.fig.canvas.draw_idle()


    # ----------------------------
    # Plot updates (fast paths)
    # ----------------------------
    def _update_main_lines(self):
        """Fast update: refresh only the green main tree in all 3 views."""
        if self.current_pc is None:
            return

        for line, dims in zip(self._main_lines, self._dims):
            if self.current_pc.size == 0:
                line.set_data([], [])
            else:
                line.set_data(self.current_pc[:, dims[0]], self.current_pc[:, dims[1]])

        self.fig.canvas.draw_idle()

    def _visible_neighbor_mask_for_view(self, view_index: int) -> np.ndarray:
        """Reduce neighbor clutter in XZ/YZ views (keeps your original logic)."""
        if self.neighbor_pc.size == 0 or self._tree_bounds is None:
            return np.zeros((self.neighbor_pc.shape[0],), dtype=bool)

        x_min, x_max, y_min, y_max = self._tree_bounds
        m = NEIGHBOR_MARGIN
        n = SIDE_MARGIN

        if view_index == 1:  # XZ front view
            return (
                # left / right
                (self.neighbor_pc[:, 0] >= x_min - m) &
                (self.neighbor_pc[:, 0] <= x_max + m) &
                # front / back → canopy-aligned
                (self.neighbor_pc[:, 1] >= y_min - n) &
                (self.neighbor_pc[:, 1] <= y_max + n)
            )

        if view_index == 2:  # YZ side view
            return (
                # left / right
                (self.neighbor_pc[:, 1] >= y_min - m) &
                (self.neighbor_pc[:, 1] <= y_max + m) &
                # front / back → canopy-aligned
                (self.neighbor_pc[:, 0] >= x_min - n) &
                (self.neighbor_pc[:, 0] <= x_max + n)
            )


        # XY: show all
        return np.ones((self.neighbor_pc.shape[0],), dtype=bool)

    def _update_neighbors_for_view(self, view_index: int, dims: Tuple[int, int], create_if_missing: bool):
        """Update (or initially create) neighbor Line2D artists for one view."""
        ax = self.axs[view_index]
        lines_by_id = self._neighbor_lines[view_index]

        if self.neighbor_pc.size == 0:
            for line in lines_by_id.values():
                line.set_data([], [])
            return

        visible_mask = self._visible_neighbor_mask_for_view(view_index)
        ids_view = self.neighbor_ids[visible_mask]
        pts_view = self.neighbor_pc[visible_mask]

        if ids_view.size == 0:
            for line in lines_by_id.values():
                line.set_data([], [])
            return

        uniq_ids, inv = np.unique(ids_view, return_inverse=True)
        uniq_set = set(int(i) for i in uniq_ids.tolist())

        for j, n_id in enumerate(uniq_ids):
            n_id = int(n_id)
            line = lines_by_id.get(n_id)

            if line is None:
                if not create_if_missing:
                    continue
                color = self._color_for_id(n_id)
                line, = ax.plot([], [], ",", color=color, alpha=0.4, zorder=1)
                lines_by_id[n_id] = line

            pts_n = pts_view[inv == j]
            line.set_data(pts_n[:, dims[0]], pts_n[:, dims[1]])

        for n_id, line in lines_by_id.items():
            if int(n_id) not in uniq_set:
                line.set_data([], [])

    def _update_neighbors_all_views(self):
        for i, dims in enumerate(self._dims):
            self._update_neighbors_for_view(view_index=i, dims=dims, create_if_missing=True)
        self.fig.canvas.draw_idle()

    # ----------------------------
    # Selection + reassignment logic
    # ----------------------------
    def on_select_with_highlight(self, verts, dims):
        if self.current_pc is None or self.current_pc.size == 0:
            return

        path = Path(verts)
        pts2d = self.current_pc[:, [dims[0], dims[1]]]
        mask = path.contains_points(pts2d)

        if not np.any(mask):
            return

        # ---- 1) SHOW HIGHLIGHT ----
        self._highlight_mask = np.zeros(len(self.current_pc), dtype=bool)
        self._highlight_mask[mask] = True
        self._update_highlight_display()

        # ---- 2) OPEN POPUP ----
        target_id = self.ask_new_id()

        # ---- 3) CLEAR HIGHLIGHT ----
        self._highlight_mask = np.zeros(len(self.current_pc), dtype=bool)
        for line in self._highlight_lines:
            line.set_data([], [])
        self.fig.canvas.draw_idle()

        if target_id is None:
            return

        # ---- 4) CONTINUE ORIGINAL LOGIC ----
        self._push_history({
            "current_pc": self.current_pc.copy(),
            "current_indices": self.current_indices.copy(),
            "reassignments": self._copy_reassignments(self.reassignments),
            "neighbor_pc": self.neighbor_pc.copy(),
            "neighbor_ids": self.neighbor_ids.copy(),
            "finalized": list(self.finalized_reassignments),
        })

        selected_indices = self.current_indices[mask].astype(int, copy=False)
        selected_coords = self.current_pc[mask].copy()

        self.reassignments.append(PendingReassignment(
            indices=selected_indices.copy(),
            new_id=int(target_id),
            coords=selected_coords,
        ))

        keep = ~mask
        self.current_pc = self.current_pc[keep]
        self.current_indices = self.current_indices[keep]

        self._update_main_lines()



    def on_select(self, verts, dims):
        if self.processing_selection:
            return
        if self.current_pc is None or self.current_pc.size == 0:
            print("on_select: nothing loaded in current_pc")
            return

        self.processing_selection = True
        try:
            path = Path(verts)
            pts2d = self.current_pc[:, [dims[0], dims[1]]]
            mask = path.contains_points(pts2d)

            n_selected = int(mask.sum())
            print(f"on_select: mask has {n_selected} points (dims={dims})")
            if n_selected == 0:
                return

            target_id = self.ask_new_id()
            if target_id is None:
                print("on_select: cancelled (no reassignment).")
                return

            self._push_history({
                "current_pc": self.current_pc.copy(),
                "current_indices": self.current_indices.copy(),
                "reassignments": self._copy_reassignments(self.reassignments),
                "neighbor_pc": self.neighbor_pc.copy(),
                "neighbor_ids": self.neighbor_ids.copy(),
                "finalized": list(self.finalized_reassignments),
            })

            selected_indices = self.current_indices[mask].astype(int, copy=False)
            selected_coords = self.current_pc[mask].copy()

            self.reassignments.append(PendingReassignment(
                indices=selected_indices.copy(),
                new_id=int(target_id),
                coords=selected_coords,
            ))
            print(f"on_select: appended reassignment, pending_count={len(self.reassignments)}")

            keep = ~mask
            self.current_pc = self.current_pc[keep]
            self.current_indices = self.current_indices[keep]

            self._update_main_lines()

        except Exception as e:
            print("Exception in on_select:", e)
            traceback.print_exc()
        finally:
            self.processing_selection = False
            self._flush_queued_actions()

    def apply_changes(self, event=None):
        """
        Commit current pending reassignments to the *visualization* (neighbor points)
        and store them in finalized_reassignments for final save.
        """
        self._push_history({
            "current_pc": self.current_pc.copy(),
            "current_indices": self.current_indices.copy(),
            "reassignments": self._copy_reassignments(self.reassignments),
            "neighbor_pc": self.neighbor_pc.copy(),
            "neighbor_ids": self.neighbor_ids.copy(),
            "finalized": list(self.finalized_reassignments),
        })
        if self.processing_selection:
            self._queue_action("apply")
            print("apply_changes: selection still processing -> queued apply.")
            return

        print("apply_changes: current pending reassignments:",
              [(int(r.indices.size), int(r.new_id)) for r in self.reassignments])

        if not self.reassignments:
            print("No changes to apply.")
            return
        self._highlight_mask[:] = False
        for line in self._highlight_lines:
            line.set_data([], [])

        self.fig.canvas.draw_idle()
        try:
            coords_chunks = []
            ids_chunks = []

            for r in self.reassignments:
                new_id = int(r.new_id)
                self.finalized_reassignments.append((r.indices.astype(int, copy=False), new_id))

                if r.coords.size > 0:
                    coords_chunks.append(r.coords)
                    ids_chunks.append(np.full((r.coords.shape[0],), new_id, dtype=int))

                self._ensure_color(new_id)

            if coords_chunks:
                new_coords = np.vstack(coords_chunks)
                new_ids = np.concatenate(ids_chunks)

                if self.neighbor_pc.size == 0:
                    self.neighbor_pc = new_coords
                    self.neighbor_ids = new_ids
                else:
                    self.neighbor_pc = np.vstack((self.neighbor_pc, new_coords))
                    self.neighbor_ids = np.concatenate((self.neighbor_ids, new_ids))

            self.reassignments = []
            

            self._update_neighbors_all_views()

            print("Applied changes to visualization (committed locally). finalized_count=",
                  len(self.finalized_reassignments))

        except Exception as e:
            print("Exception in apply_changes:", e)
            traceback.print_exc()

    # ----------------------------
    # Undo / Erase
    # ----------------------------
    def undo(self, event=None):
        if self.processing_selection:
            self._queue_action("undo")
            print("undo: selection still processing -> queued undo.")
            return

        if not self.history:
            return

        state = self.history.pop()

        self.current_pc = state["current_pc"]
        self.current_indices = state["current_indices"]
        self.reassignments = self._copy_reassignments(state["reassignments"])
        self.neighbor_pc = state["neighbor_pc"]
        self.neighbor_ids = state["neighbor_ids"]
        self.finalized_reassignments = list(state["finalized"])

        self._update_main_lines()
        self._update_neighbors_all_views()
        self._highlight_mask[:] = False
        for line in self._highlight_lines:
            line.set_data([], [])

        self.fig.canvas.draw_idle()

        print("Undo restored full previous state.")

    def erase_tree(self, event=None):
        if self.processing_selection:
            self._queue_action("erase")
            print("erase_tree: selection still processing -> queued erase.")
            return

        if self.current_indices is None or self.current_indices.size == 0:
            return

        self._push_history({
            "current_pc": self.current_pc.copy(),
            "current_indices": self.current_indices.copy(),
            "reassignments": self._copy_reassignments(self.reassignments),
            "neighbor_pc": self.neighbor_pc.copy(),
            "neighbor_ids": self.neighbor_ids.copy(),
            "finalized": list(self.finalized_reassignments),
        })

        self.reassignments.append(PendingReassignment(
            indices=self.current_indices.copy(),
            new_id=NOISE_ID,
            coords=self.current_pc.copy(),
        ))

        self.current_pc = np.empty((0, 3), dtype=float)
        self.current_indices = np.empty((0,), dtype=int)

        self._update_main_lines()
        print("Entire tree marked for reclassification to", NOISE_ID)

    def _push_history(self, state):
        self.history.append(state)

        # limit history size
        if len(self.history) > self.max_history:
            self.history.pop(0)

    # ----------------------------
    # Save/Skip/Exit
    # ----------------------------
    def _disable_selectors(self):
        for sel in self._selectors:
            try:
                sel.set_active(False)
            except Exception:
                pass

    def save_exit(self, event=None):
        if self.processing_selection:
            self._queue_action("save")
            print("save_exit: selection still processing -> queued save.")
            return

        self._disable_selectors()
        self.save_requested = True
        plt.close(self.fig)

    def skip_exit(self, event=None):
        if self.processing_selection:
            self._queue_action("skip")
            print("skip_exit: selection still processing -> queued skip.")
            return

        self._disable_selectors()
        self.save_requested = False
        plt.close(self.fig)

    def stop_all(self, event=None):
        if self.processing_selection:
            self._queue_action("stop")
            print("stop_all: selection still processing -> queued stop.")
            return

        self._disable_selectors()
        self.stop_flag = True
        plt.close(self.fig)

    def _on_close(self, event):
        self._disable_selectors()

    # ----------------------------
    # Mouse wheel zoom
    # ----------------------------
    def on_scroll(self, event):
        if event.inaxes is None:
            return

        ax = event.inaxes
        base_scale = 1.2

        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

        scale_factor = 1 / base_scale if event.button == "up" else base_scale
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        rel_x = (cur_xlim[1] - event.xdata) / (cur_xlim[1] - cur_xlim[0])
        rel_y = (cur_ylim[1] - event.ydata) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([event.xdata - new_width * (1 - rel_x), event.xdata + new_width * rel_x])
        ax.set_ylim([event.ydata - new_height * (1 - rel_y), event.ydata + new_height * rel_y])

        self.fig.canvas.draw_idle()

    def ask_new_id(self) -> Optional[int]:
        selectable_ids = [nid for nid in sorted(self.id_colors.keys())
                        if nid not in (NOISE_ID, UNCLASSIFIED_ID)]

        result = {"id": None}

        dlg = QDialog()
        dlg.setWindowTitle("Reassign Points")
        dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout()
        dlg.setLayout(layout)
        layout.addWidget(QLabel("<b>Select Target Tree ID:</b>"))

        def select(val):
            result["id"] = val
            dlg.accept()

        def select_new_tree():
            new_id = self.next_new_tree_id
            self.next_new_tree_id += 1
            result["id"] = new_id
            dlg.accept()

        btn_noise = QPushButton(f"Noise ({NOISE_ID})")
        btn_noise.setStyleSheet("background-color: #ffcccc;")
        btn_noise.clicked.connect(lambda: select(NOISE_ID))
        layout.addWidget(btn_noise)

        btn_new = QPushButton(f"➕ New Tree (ID {self.next_new_tree_id})")
        btn_new.setStyleSheet("background-color: #ccffcc;")
        btn_new.clicked.connect(select_new_tree)
        layout.addWidget(btn_new)

        if not selectable_ids:
            layout.addWidget(QLabel("(No neighbor trees found nearby)"))
        else:
            for nid in selectable_ids:
                color = self.id_colors.get(nid, "#aaaaaa")
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)

                swatch = QFrame()
                swatch.setFixedSize(15, 15)
                swatch.setStyleSheet(f"background-color: {color};")
                row_layout.addWidget(swatch)

                btn = QPushButton(f"Tree ID: {nid}")
                btn.clicked.connect(lambda checked, n=int(nid): select(n))
                row_layout.addWidget(btn)

                layout.addWidget(row)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(dlg.reject)
        layout.addWidget(btn_cancel)

        dlg.exec_()  # blocking modal — replaces dlg.wait_window()
        return result["id"]
    

    # ----------------------------
    # Colors
    # ----------------------------
    def _palette(self) -> list[str]:
        cmaps = [plt.get_cmap("tab20"), plt.get_cmap("tab20b"), plt.get_cmap("tab20c")]
        colors = []
        for cmap in cmaps:
            for i in range(cmap.N):
                colors.append(mcolors.to_hex(cmap(i)))

        forbidden = {4, 5}  # optional
        colors = [c for idx, c in enumerate(colors) if idx not in forbidden]
        return colors


    def _init_id_colors(self, current_tid: int):
        self.id_colors = {}

        neighbor_unique = np.unique(self.neighbor_ids) if self.neighbor_ids.size else np.array([], dtype=int)
        neighbor_unique = neighbor_unique[(neighbor_unique != UNCLASSIFIED_ID) & (neighbor_unique != current_tid)]

        pal = self._palette()  

        for idx, nid in enumerate(sorted(int(n) for n in neighbor_unique.tolist() if int(n) != NOISE_ID)):
            self.id_colors[int(nid)] = pal[idx % len(pal)]

        self.id_colors[NOISE_ID] = "#FFFFFF"

    def _ensure_color(self, n_id: int):
        if n_id in self.id_colors:
            return
        if n_id == NOISE_ID:
            self.id_colors[n_id] = "#FFFFFF"
            return

        pal = self._palette()
        used = {c for k, c in self.id_colors.items() if k != NOISE_ID}
        for c in pal:
            if c not in used:
                self.id_colors[n_id] = c
                return
        self.id_colors[n_id] = pal[len(used) % len(pal)]

    def _color_for_id(self, n_id: int) -> str:
        self._ensure_color(int(n_id))
        return self.id_colors.get(int(n_id), "#aaaaaa")

    # ----------------------------
    # Queued action handling (fixes the Apply/lasso race)
    # ----------------------------
    def _queue_action(self, action: str):
        if action not in self._queued_actions:
            self._queued_actions.append(action)

    def _flush_queued_actions(self):
        if not self._queued_actions:
            return

        actions = self._queued_actions
        self._queued_actions = []

        for a in actions:
            try:
                if a == "apply":
                    self.apply_changes(None)
                elif a == "save":
                    self.save_exit(None)
                elif a == "skip":
                    self.skip_exit(None)
                elif a == "stop":
                    self.stop_all(None)
                elif a == "undo":
                    self.undo(None)
                elif a == "erase":
                    self.erase_tree(None)
            except Exception:
                traceback.print_exc()

    # ----------------------------
    # Helpers
    # ----------------------------
    @staticmethod
    def _copy_reassignments(reassignments: List[PendingReassignment]) -> List[PendingReassignment]:
        return [
            PendingReassignment(
                indices=r.indices.copy(),
                new_id=int(r.new_id),
                coords=r.coords.copy(),
            )
            for r in reassignments
        ]


if __name__ == "__main__":
    inspector = TreeInspector(tree_dir = "C:/ICEsat_Project/TLS_plot_for_testing",
                              out_dir="C:/ICEsat_Project/TLS_plot_for_testing/done")
    
    inspector.run()
    
