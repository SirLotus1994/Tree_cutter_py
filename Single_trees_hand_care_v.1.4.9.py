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

from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QWidget, QLineEdit,
    QFileDialog, QFormLayout, QMessageBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# ----------------------------
# Configuration
# ----------------------------
NEIGHBOR_MARGIN = 3.0     # meters around the current tree bbox to show neighbors
SIDE_MARGIN     = 0.5
NOISE_ID        = 999     # points reclassified as noise
UNCLASSIFIED_ID = 0       # points that are not assigned to any tree


# ============================================================
#  Launch dialog
# ============================================================
class LaunchDialog(QDialog):
    """
    Startup dialog that collects:
      • Input folder  (folder of .laz files)
      • Output folder (where *_updated.laz files are saved)
      • ID field name (the LAS extra-dim that holds the tree/instance ID,
                       e.g. 'PredInstance', 'treeID', …)
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tree Cutter — Launch")
        self.setMinimumWidth(520)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        # ----- result values -----
        self.input_dir:  Optional[str] = None
        self.output_dir: Optional[str] = None
        self.id_field:   str = "PredInstance"

        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)

        # Title
        title = QLabel("🌲  Tree Cutter")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        subtitle = QLabel("Configure input, output, and ID field before starting.")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: gray;")
        root.addWidget(subtitle)

        root.addSpacing(6)

        # Form
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(10)

        # — Input folder —
        self._in_edit  = QLineEdit()
        self._in_edit.setPlaceholderText("Folder containing .laz files…")
        in_row = self._folder_row(self._in_edit, self._browse_input)
        form.addRow("Input folder:", in_row)

        # — Output folder —
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("Leave blank to use input folder")
        out_row = self._folder_row(self._out_edit, self._browse_output)
        form.addRow("Output folder:", out_row)

        # — ID field —
        self._field_edit = QLineEdit("PredInstance")
        self._field_edit.setToolTip(
            "Name of the LAS extra dimension that stores the tree / instance ID.\n"
            "Common values: PredInstance, treeID, instance_id …"
        )
        form.addRow("ID field name:", self._field_edit)

        root.addLayout(form)
        root.addSpacing(10)

        # — Buttons —
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        btn_run = QPushButton("▶  Start Refinement")
        btn_run.setDefault(True)
        btn_run.setStyleSheet(
            "background-color: #2ecc71; color: white; font-weight: bold; padding: 6px 16px;"
        )
        btn_run.clicked.connect(self._on_start)
        btn_row.addWidget(btn_run)

        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    @staticmethod
    def _folder_row(edit: QLineEdit, browse_fn) -> QWidget:
        w   = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(edit)
        btn = QPushButton("Browse…")
        btn.setFixedWidth(80)
        btn.clicked.connect(browse_fn)
        lay.addWidget(btn)
        return w

    def _browse_input(self):
        d = QFileDialog.getExistingDirectory(self, "Select Input Folder", "")
        if d:
            self._in_edit.setText(d)
            # Pre-fill output if still empty
            if not self._out_edit.text().strip():
                self._out_edit.setText(os.path.join(d, "done"))

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Folder", "")
        if d:
            self._out_edit.setText(d)

    # ------------------------------------------------------------------
    def _on_start(self):
        in_dir    = self._in_edit.text().strip()
        out_dir   = self._out_edit.text().strip()
        id_field  = self._field_edit.text().strip()

        # Validation
        if not in_dir:
            QMessageBox.warning(self, "Missing input", "Please select an input folder.")
            return
        if not os.path.isdir(in_dir):
            QMessageBox.warning(self, "Invalid input", f"Folder not found:\n{in_dir}")
            return
        if not id_field:
            QMessageBox.warning(self, "Missing field", "Please enter the ID field name.")
            return

        # Output folder: create if needed
        if not out_dir:
            out_dir = in_dir
        os.makedirs(out_dir, exist_ok=True)

        self.input_dir  = in_dir
        self.output_dir = out_dir
        self.id_field   = id_field
        self.accept()


# ============================================================
#  Data classes
# ============================================================
@dataclass
class PendingReassignment:
    """A not-yet-applied reassignment (kept until 'Apply & Stay' or final save)."""
    indices: np.ndarray   # global indices in the LAS/LAZ file
    new_id:  int          # target tree ID (or NOISE_ID)
    coords:  np.ndarray   # XYZ coordinates of the moved points (for visualization)


# ============================================================
#  Main inspector class
# ============================================================
class TreeInspector:
    def __init__(self, tree_dir: str, out_dir: Optional[str] = None, id_field: str = "PredInstance"):
        self.tree_dir = tree_dir
        self.out_dir  = out_dir if out_dir is not None else tree_dir
        self.id_field = id_field
        self.next_new_tree_id: Optional[int] = None

        # Control flags
        self.stop_flag:           bool = False
        self.save_requested:      bool = False
        self.processing_selection: bool = False
        self._queued_actions:    List[str] = []

        # Current tree state
        self.current_pc:       Optional[np.ndarray] = None
        self.current_indices:  Optional[np.ndarray] = None
        self._original_tree_indices: Optional[np.ndarray] = None

        # Neighbor state
        self.neighbor_pc:  np.ndarray = np.empty((0, 3), dtype=float)
        self.neighbor_ids: np.ndarray = np.empty((0,),   dtype=int)

        # Undo stack
        self.history:    List[dict] = []
        self.max_history = 20

        # Reassignment buffers
        self.reassignments:           List[PendingReassignment]    = []
        self.finalized_reassignments: List[Tuple[np.ndarray, int]] = []
        self._highlight_mask = None

        # Color mapping
        self.id_colors: Dict[int, str] = {}

        # Matplotlib artists
        self.fig           = None
        self.axs           = None
        self._selectors    = []
        self._main_lines   = []
        self._neighbor_lines: List[Dict[int, object]] = []

        self._tree_bounds  = None
        self._qapp = QApplication.instance() or QApplication(sys.argv)

    # ----------------------------
    # Main loop
    # ----------------------------
    def run(self):
        laz_files = sorted(glob.glob(os.path.join(self.tree_dir, "*.laz")))
        if not laz_files:
            print(f"No .laz files found in: {self.tree_dir}")
            return

        for f_path in laz_files:
            if self.stop_flag:
                break

            print(f"\nProcessing {os.path.basename(f_path)}")
            las = laspy.read(f_path)

            # Read the user-specified ID field
            try:
                tree_ids = np.asarray(getattr(las, self.id_field)).copy()
            except AttributeError:
                print(f"  ⚠  Field '{self.id_field}' not found in {os.path.basename(f_path)}. Skipping.")
                available = [str(d.name) for d in las.point_format.dimensions]
                print(f"     Available fields: {available}")
                continue

            x = np.asarray(las.x)
            y = np.asarray(las.y)
            z = np.asarray(las.z)

            unique_ids = np.unique(tree_ids)
            unique_ids = unique_ids[(unique_ids > 0) & (unique_ids != NOISE_ID)]
            unique_ids = np.sort(unique_ids)
            self.next_new_tree_id = int(unique_ids.max() + 1) if unique_ids.size else 1
            print("Trees:", unique_ids)

            for tid in unique_ids:
                if self.stop_flag:
                    break

                mask_main             = (tree_ids == tid)
                original_tree_indices = np.flatnonzero(mask_main)
                if original_tree_indices.size == 0:
                    continue

                self._original_tree_indices = original_tree_indices
                self.current_pc      = np.column_stack((x[mask_main], y[mask_main], z[mask_main])).astype(float, copy=False)
                self.current_indices = original_tree_indices.copy()

                x_min, x_max = float(self.current_pc[:, 0].min()), float(self.current_pc[:, 0].max())
                y_min, y_max = float(self.current_pc[:, 1].min()), float(self.current_pc[:, 1].max())
                self._tree_bounds = (x_min, x_max, y_min, y_max)

                m = NEIGHBOR_MARGIN
                mask_neighbors = (
                    (tree_ids != tid)
                    & (tree_ids != UNCLASSIFIED_ID)
                    & (x >= (x_min - m)) & (x <= (x_max + m))
                    & (y >= (y_min - m)) & (y <= (y_max + m))
                )

                if np.any(mask_neighbors):
                    self.neighbor_pc  = np.column_stack((x[mask_neighbors], y[mask_neighbors], z[mask_neighbors])).astype(float, copy=False)
                    self.neighbor_ids = tree_ids[mask_neighbors].astype(int, copy=False)
                else:
                    self.neighbor_pc  = np.empty((0, 3), dtype=float)
                    self.neighbor_ids = np.empty((0,),   dtype=int)

                # Reset per-tree buffers
                self.reassignments           = []
                self.finalized_reassignments = []
                self.history                 = []
                self.save_requested          = False
                self.processing_selection    = False
                self._queued_actions         = []

                self.open_gui(f_path, int(tid))

                print(
                    "AFTER GUI close: len(reassignments) =", len(self.reassignments),
                    "len(finalized_reassignments) =", len(self.finalized_reassignments),
                )

                if self.save_requested:
                    to_apply: List[Tuple[np.ndarray, int]] = []
                    to_apply.extend(self.finalized_reassignments)
                    to_apply.extend((r.indices, int(r.new_id)) for r in self.reassignments)

                    for idx_list, new_id in to_apply:
                        tree_ids[idx_list] = new_id

                    if self._original_tree_indices is not None and self.current_indices is not None:
                        removed = np.setdiff1d(self._original_tree_indices, self.current_indices, assume_unique=False)
                        if to_apply:
                            reassigned_flat = np.concatenate([idx for idx, _ in to_apply]).astype(int, copy=False)
                        else:
                            reassigned_flat = np.empty((0,), dtype=int)
                        remaining_removed = np.setdiff1d(removed, reassigned_flat, assume_unique=False)
                        if remaining_removed.size > 0:
                            tree_ids[remaining_removed] = NOISE_ID

            # Write back using the same field name
            setattr(las, self.id_field, tree_ids)
            base     = os.path.splitext(os.path.basename(f_path))[0]
            out_path = os.path.join(self.out_dir, f"{base}_updated.laz")
            las.write(out_path)
            print(f"Saved: {out_path}")

    # ----------------------------
    # GUI building
    # ----------------------------
    def open_gui(self, filename: str, tid: int):
        self._init_id_colors(tid)

        self.fig, self.axs = plt.subplots(1, 3, figsize=(18, 7), facecolor="black")
        self.fig.canvas.manager.set_window_title(
            f"Tree ID {tid} | {os.path.basename(filename)} | field: {self.id_field} | Scroll to Zoom"
        )

        titles    = ["XY Top View", "XZ Front View", "YZ Side View"]
        self._dims = [(0, 1), (0, 2), (1, 2)]

        self._selectors       = []
        self._main_lines      = []
        self._highlight_lines = []
        self._neighbor_lines  = [dict(), dict(), dict()]
        self._highlight_mask  = np.zeros(len(self.current_pc), dtype=bool)

        for i, (ax, title, dims) in enumerate(zip(self.axs, titles, self._dims)):
            ax.set_facecolor("black")
            ax.grid(True, color="#222222", linestyle="--", linewidth=0.5)
            ax.tick_params(colors="white", labelsize=7)
            ax.set_title(title, color="white", fontsize=10)

            self._update_neighbors_for_view(view_index=i, dims=dims, create_if_missing=True)

            main_line, = ax.plot(
                self.current_pc[:, dims[0]] if self.current_pc is not None else [],
                self.current_pc[:, dims[1]] if self.current_pc is not None else [],
                ",", color="#00FF00", alpha=0.7, zorder=2,
            )

            highlight_line, = ax.plot([], [], ",", color="#FFFF00", alpha=1.0, zorder=5)
            self._highlight_lines.append(highlight_line)
            self._main_lines.append(main_line)

            ax.set_aspect("equal", "datalim")

            try:
                selector = LassoSelector(
                    ax,
                    onselect=lambda verts, d=dims: self.on_select_with_highlight(verts, d),
                    useblit=True,
                )
            except TypeError:
                selector = LassoSelector(ax, onselect=self.on_select_with_highlight(dims))
            self._selectors.append(selector)

        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self._setup_buttons()
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        plt.show()

    def _setup_buttons(self):
        plt.subplots_adjust(bottom=0.2)

        ax_undo  = plt.axes([0.05, 0.05, 0.08, 0.05])
        ax_apply = plt.axes([0.15, 0.05, 0.15, 0.05])
        ax_erase = plt.axes([0.32, 0.05, 0.10, 0.05])
        ax_save  = plt.axes([0.44, 0.05, 0.15, 0.05])
        ax_skip  = plt.axes([0.61, 0.05, 0.10, 0.05])
        ax_stop  = plt.axes([0.75, 0.05, 0.10, 0.05])

        self.btn_undo  = Button(ax_undo,  "Undo")
        self.btn_apply = Button(ax_apply, "Apply & Stay", color="#3498db", hovercolor="#2980b9")
        self.btn_erase = Button(ax_erase, "Erase Tree")
        self.btn_save  = Button(ax_save,  "Save & Next",  color="#2ecc71")
        self.btn_skip  = Button(ax_skip,  "Skip")
        self.btn_stop  = Button(ax_stop,  "Exit")

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
            self._highlight_lines[i].set_data(pts[:, dims[0]], pts[:, dims[1]])
        self.fig.canvas.draw_idle()

    # ----------------------------
    # Plot updates
    # ----------------------------
    def _update_main_lines(self):
        if self.current_pc is None:
            return
        for line, dims in zip(self._main_lines, self._dims):
            if self.current_pc.size == 0:
                line.set_data([], [])
            else:
                line.set_data(self.current_pc[:, dims[0]], self.current_pc[:, dims[1]])
        self.fig.canvas.draw_idle()

    def _visible_neighbor_mask_for_view(self, view_index: int) -> np.ndarray:
        if self.neighbor_pc.size == 0 or self._tree_bounds is None:
            return np.zeros((self.neighbor_pc.shape[0],), dtype=bool)

        x_min, x_max, y_min, y_max = self._tree_bounds
        m = NEIGHBOR_MARGIN
        n = SIDE_MARGIN

        if view_index == 1:  # XZ front
            return (
                (self.neighbor_pc[:, 0] >= x_min - m) & (self.neighbor_pc[:, 0] <= x_max + m) &
                (self.neighbor_pc[:, 1] >= y_min - n) & (self.neighbor_pc[:, 1] <= y_max + n)
            )
        if view_index == 2:  # YZ side
            return (
                (self.neighbor_pc[:, 1] >= y_min - m) & (self.neighbor_pc[:, 1] <= y_max + m) &
                (self.neighbor_pc[:, 0] >= x_min - n) & (self.neighbor_pc[:, 0] <= x_max + n)
            )
        return np.ones((self.neighbor_pc.shape[0],), dtype=bool)

    def _update_neighbors_for_view(self, view_index: int, dims: Tuple[int, int], create_if_missing: bool):
        ax         = self.axs[view_index]
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
    # Selection + reassignment
    # ----------------------------
    def on_select_with_highlight(self, verts, dims):
        if self.current_pc is None or self.current_pc.size == 0:
            return

        path = Path(verts)
        pts2d = self.current_pc[:, [dims[0], dims[1]]]
        mask  = path.contains_points(pts2d)
        if not np.any(mask):
            return

        self._highlight_mask = np.zeros(len(self.current_pc), dtype=bool)
        self._highlight_mask[mask] = True
        self._update_highlight_display()

        target_id = self.ask_new_id()

        self._highlight_mask = np.zeros(len(self.current_pc), dtype=bool)
        for line in self._highlight_lines:
            line.set_data([], [])
        self.fig.canvas.draw_idle()

        if target_id is None:
            return

        self._push_history({
            "current_pc":      self.current_pc.copy(),
            "current_indices": self.current_indices.copy(),
            "reassignments":   self._copy_reassignments(self.reassignments),
            "neighbor_pc":     self.neighbor_pc.copy(),
            "neighbor_ids":    self.neighbor_ids.copy(),
            "finalized":       list(self.finalized_reassignments),
        })

        selected_indices = self.current_indices[mask].astype(int, copy=False)
        selected_coords  = self.current_pc[mask].copy()

        self.reassignments.append(PendingReassignment(
            indices=selected_indices.copy(),
            new_id=int(target_id),
            coords=selected_coords,
        ))

        keep = ~mask
        self.current_pc      = self.current_pc[keep]
        self.current_indices = self.current_indices[keep]
        self._update_main_lines()

    def apply_changes(self, event=None):
        self._push_history({
            "current_pc":      self.current_pc.copy(),
            "current_indices": self.current_indices.copy(),
            "reassignments":   self._copy_reassignments(self.reassignments),
            "neighbor_pc":     self.neighbor_pc.copy(),
            "neighbor_ids":    self.neighbor_ids.copy(),
            "finalized":       list(self.finalized_reassignments),
        })
        if self.processing_selection:
            self._queue_action("apply")
            return

        if not self.reassignments:
            return

        self._highlight_mask[:] = False
        for line in self._highlight_lines:
            line.set_data([], [])
        self.fig.canvas.draw_idle()

        try:
            coords_chunks, ids_chunks = [], []
            for r in self.reassignments:
                new_id = int(r.new_id)
                self.finalized_reassignments.append((r.indices.astype(int, copy=False), new_id))
                if r.coords.size > 0:
                    coords_chunks.append(r.coords)
                    ids_chunks.append(np.full((r.coords.shape[0],), new_id, dtype=int))
                self._ensure_color(new_id)

            if coords_chunks:
                new_coords = np.vstack(coords_chunks)
                new_ids    = np.concatenate(ids_chunks)
                if self.neighbor_pc.size == 0:
                    self.neighbor_pc  = new_coords
                    self.neighbor_ids = new_ids
                else:
                    self.neighbor_pc  = np.vstack((self.neighbor_pc, new_coords))
                    self.neighbor_ids = np.concatenate((self.neighbor_ids, new_ids))

            self.reassignments = []
            self._update_neighbors_all_views()
        except Exception as e:
            print("Exception in apply_changes:", e)
            traceback.print_exc()

    # ----------------------------
    # Undo / Erase
    # ----------------------------
    def undo(self, event=None):
        if self.processing_selection:
            self._queue_action("undo")
            return
        if not self.history:
            return

        state = self.history.pop()
        self.current_pc              = state["current_pc"]
        self.current_indices         = state["current_indices"]
        self.reassignments           = self._copy_reassignments(state["reassignments"])
        self.neighbor_pc             = state["neighbor_pc"]
        self.neighbor_ids            = state["neighbor_ids"]
        self.finalized_reassignments = list(state["finalized"])

        self._update_main_lines()
        self._update_neighbors_all_views()
        self._highlight_mask[:] = False
        for line in self._highlight_lines:
            line.set_data([], [])
        self.fig.canvas.draw_idle()

    def erase_tree(self, event=None):
        if self.processing_selection:
            self._queue_action("erase")
            return
        if self.current_indices is None or self.current_indices.size == 0:
            return

        self._push_history({
            "current_pc":      self.current_pc.copy(),
            "current_indices": self.current_indices.copy(),
            "reassignments":   self._copy_reassignments(self.reassignments),
            "neighbor_pc":     self.neighbor_pc.copy(),
            "neighbor_ids":    self.neighbor_ids.copy(),
            "finalized":       list(self.finalized_reassignments),
        })

        self.reassignments.append(PendingReassignment(
            indices=self.current_indices.copy(),
            new_id=NOISE_ID,
            coords=self.current_pc.copy(),
        ))
        self.current_pc      = np.empty((0, 3), dtype=float)
        self.current_indices = np.empty((0,),   dtype=int)
        self._update_main_lines()

    def _push_history(self, state):
        self.history.append(state)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    # ----------------------------
    # Save / Skip / Exit
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
            return
        self._disable_selectors()
        self.save_requested = True
        plt.close(self.fig)

    def skip_exit(self, event=None):
        if self.processing_selection:
            self._queue_action("skip")
            return
        self._disable_selectors()
        self.save_requested = False
        plt.close(self.fig)

    def stop_all(self, event=None):
        if self.processing_selection:
            self._queue_action("stop")
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
        ax         = event.inaxes
        base_scale = 1.2
        cur_xlim   = ax.get_xlim()
        cur_ylim   = ax.get_ylim()
        scale_factor = 1 / base_scale if event.button == "up" else base_scale
        new_width    = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height   = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        rel_x = (cur_xlim[1] - event.xdata) / (cur_xlim[1] - cur_xlim[0])
        rel_y = (cur_ylim[1] - event.ydata) / (cur_ylim[1] - cur_ylim[0])
        ax.set_xlim([event.xdata - new_width * (1 - rel_x), event.xdata + new_width * rel_x])
        ax.set_ylim([event.ydata - new_height * (1 - rel_y), event.ydata + new_height * rel_y])
        self.fig.canvas.draw_idle()

    # ----------------------------
    # Ask target ID dialog
    # ----------------------------
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
                row   = QWidget()
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

        dlg.exec_()
        return result["id"]

    # ----------------------------
    # Colors
    # ----------------------------
    def _palette(self) -> list:
        cmaps = [plt.get_cmap("tab20"), plt.get_cmap("tab20b"), plt.get_cmap("tab20c")]
        colors = []
        for cmap in cmaps:
            for i in range(cmap.N):
                colors.append(mcolors.to_hex(cmap(i)))
        forbidden = {4, 5}
        return [c for idx, c in enumerate(colors) if idx not in forbidden]

    def _init_id_colors(self, current_tid: int):
        self.id_colors   = {}
        neighbor_unique  = np.unique(self.neighbor_ids) if self.neighbor_ids.size else np.array([], dtype=int)
        neighbor_unique  = neighbor_unique[(neighbor_unique != UNCLASSIFIED_ID) & (neighbor_unique != current_tid)]
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
        pal  = self._palette()
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
    # Queued actions
    # ----------------------------
    def _queue_action(self, action: str):
        if action not in self._queued_actions:
            self._queued_actions.append(action)

    def _flush_queued_actions(self):
        if not self._queued_actions:
            return
        actions = self._queued_actions[:]
        self._queued_actions = []
        for a in actions:
            try:
                if   a == "apply": self.apply_changes(None)
                elif a == "save":  self.save_exit(None)
                elif a == "skip":  self.skip_exit(None)
                elif a == "stop":  self.stop_all(None)
                elif a == "undo":  self.undo(None)
                elif a == "erase": self.erase_tree(None)
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


# ============================================================
#  Entry point
# ============================================================
if __name__ == "__main__":
    qapp = QApplication.instance() or QApplication(sys.argv)

    # Show launch dialog first
    dlg = LaunchDialog()
    if dlg.exec_() != QDialog.Accepted:
        print("Launch cancelled.")
        sys.exit(0)

    inspector = TreeInspector(
        tree_dir  = dlg.input_dir,
        out_dir   = dlg.output_dir,
        id_field  = dlg.id_field,
    )
    inspector.run()