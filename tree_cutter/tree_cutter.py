
# Tree Inspector - pyqtgraph.opengl 1.6.0


import os
import sys
import glob
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

from PyQt5.QtCore import Qt, QCoreApplication
QCoreApplication.setAttribute(Qt.AA_UseDesktopOpenGL)
QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

import numpy as np
import laspy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.path import Path

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, QPoint
from PyQt5.QtGui import (
    QFont, QColor, QMatrix4x4, QVector3D, QPainter, QPen, QPolygon,
    QFontMetrics,
)
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFrame, QWidget, QLineEdit,
    QFileDialog, QFormLayout, QMessageBox,
)

try:
    from OpenGL.GL import (
        GL_DEPTH_TEST, GL_BLEND, GL_ALPHA_TEST, GL_CULL_FACE,
        GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    )
    SCATTER_GL_OPTIONS = {
        GL_DEPTH_TEST:   False,
        GL_BLEND:        True,
        GL_ALPHA_TEST:   False,
        GL_CULL_FACE:    False,
        'glBlendFunc':   (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA),
    }
except ImportError:
    # PyOpenGL missing: fall back to pyqtgraph's named preset. Depth
    # test will be on; neighbour colours may still look funny but at
    # least the app will run.
    SCATTER_GL_OPTIONS = 'translucent'


# ----------------------------
# Configuration
# ----------------------------
NEIGHBOR_MARGIN = 3.0
SIDE_MARGIN     = 0.5
NOISE_ID        = 999
UNCLASSIFIED_ID = 0

POINT_SIZE_PX = 2.0
NEIGHBOR_ALPHA = 0.85

# Pixels reserved at the left edge of each panel for the y-axis tick
# labels (when present). When the y-axis is hidden (top view) we don't
# need this, but keeping the value uniform keeps the three panels lined
# up vertically.
LEFT_AXIS_PAD_PX   = 40
BOTTOM_AXIS_PAD_PX = 22
TICK_LEN_PX        = 5


# ============================================================
#  Launch dialog
# ============================================================
class LaunchDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tree Cutter Py - Launch")
        self.setMinimumWidth(520)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        self.input_dir:  Optional[str] = None
        self.output_dir: Optional[str] = None
        self.id_field:   str = "PredInstance"
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)

        title = QLabel("Tree Cutter Py")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        subtitle = QLabel("The software will go through all files in the directory")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: gray;")
        root.addWidget(subtitle)
        root.addSpacing(6)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(10)

        self._in_edit = QLineEdit()
        self._in_edit.setPlaceholderText("Folder containing .laz files...")
        form.addRow("Input folder:", self._folder_row(self._in_edit, self._browse_input))

        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("Leave blank to use input folder")
        form.addRow("Output folder:", self._folder_row(self._out_edit, self._browse_output))

        self._field_edit = QLineEdit("PredInstance")
        form.addRow("ID field name:", self._field_edit)
        root.addLayout(form)
        root.addSpacing(10)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_cancel = QPushButton("Cancel"); btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        btn_run = QPushButton("Start Refinement")
        btn_run.setDefault(True)
        btn_run.setStyleSheet(
            "background-color: #2ecc71; color: white; font-weight: bold; padding: 6px 16px;"
        )
        btn_run.clicked.connect(self._on_start)
        btn_row.addWidget(btn_run)
        root.addLayout(btn_row)

    @staticmethod
    def _folder_row(edit, browse_fn):
        w = QWidget()
        lay = QHBoxLayout(w); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(6)
        lay.addWidget(edit)
        btn = QPushButton("Browse..."); btn.setFixedWidth(80); btn.clicked.connect(browse_fn)
        lay.addWidget(btn)
        return w

    def _browse_input(self):
        d = QFileDialog.getExistingDirectory(self, "Select Input Folder", "")
        if d:
            self._in_edit.setText(d)
            if not self._out_edit.text().strip():
                self._out_edit.setText(os.path.join(d, "done"))

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Folder", "")
        if d:
            self._out_edit.setText(d)

    def _on_start(self):
        in_dir   = self._in_edit.text().strip()
        out_dir  = self._out_edit.text().strip()
        id_field = self._field_edit.text().strip()
        if not in_dir:
            QMessageBox.warning(self, "Missing input", "Please select an input folder.")
            return
        if not os.path.isdir(in_dir):
            QMessageBox.warning(self, "Invalid input", f"Folder not found:\n{in_dir}")
            return
        if not id_field:
            QMessageBox.warning(self, "Missing field", "Please enter the ID field name.")
            return
        if not out_dir:
            out_dir = in_dir
        os.makedirs(out_dir, exist_ok=True)
        self.input_dir = in_dir
        self.output_dir = out_dir
        self.id_field   = id_field
        self.accept()


# ============================================================
#  Data classes
# ============================================================
@dataclass
class PendingReassignment:
    indices: np.ndarray
    new_id:  int
    coords:  np.ndarray


# ============================================================
#  Helpers
# ============================================================
def _nice_ticks(a: float, b: float, target_count: int = 5,
                integer_only: bool = False) -> List[float]:
    """Pick reasonably round tick values in the range [a, b]."""
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        return []
    rng = b - a
    raw_step = rng / max(target_count, 1)
    if raw_step <= 0:
        return []
    magnitude = 10 ** int(np.floor(np.log10(raw_step)))
    candidates = [1, 2, 5, 10]
    step = magnitude
    for c in candidates:
        if c * magnitude >= raw_step:
            step = c * magnitude
            break
    if integer_only:
        step = max(1.0, float(int(round(step))))
    # First tick at or above a.
    first = np.ceil(a / step) * step
    ticks = []
    t = first
    # Tolerance to include endpoint.
    while t <= b + step * 1e-6:
        ticks.append(round(t, 9))
        t += step
    if integer_only:
        ticks = sorted({int(round(t)) for t in ticks})
    return ticks


# ============================================================
#  OrthoGLView - one panel
# ============================================================
class OrthoGLView(gl.GLViewWidget):
    """Top-down orthographic GL panel. Pan = right drag, zoom = wheel
    (around cursor), lasso = left drag (with live yellow preview)."""

    sigLassoFinished = QtCore.pyqtSignal(object, int)   # (world verts, view_idx)

    def __init__(
        self,
        dims: Tuple[int, int],
        view_idx: int,
        show_y_labels: bool = True,
        integer_y: bool = False,
        x_label: str = "X",
        y_label: str = "Y",
        parent=None,
    ):
        super().__init__(parent)
        self.dims = dims
        self.view_idx = view_idx
        self.show_y_labels = show_y_labels
        self.integer_y = integer_y
        self.x_label = x_label
        self.y_label = y_label

        self.ortho_scale = 50.0
        self.opts['azimuth']    = -90.0
        self.opts['elevation']  = 90.0
        self.opts['fov']        = 60.0
        self.opts['distance']   = 1.0
        self.opts['center']     = QVector3D(0.0, 0.0, 0.0)

        # No right-click context menu (it intercepts button events).
        self.setContextMenuPolicy(Qt.NoContextMenu)

        self._pan_active   = False
        self._pan_last     = (0, 0)
        self._lasso_active = False
        self._lasso_pts: List[Tuple[int, int]] = []

        self.neighbor_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=np.zeros((0, 4), dtype=np.float32),
            size=POINT_SIZE_PX,
            pxMode=True,
            glOptions=SCATTER_GL_OPTIONS,
        )
        self.neighbor_scatter.setDepthValue(0)
        self.addItem(self.neighbor_scatter)

        self.main_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=np.zeros((0, 4), dtype=np.float32),
            size=POINT_SIZE_PX,
            pxMode=True,
            glOptions=SCATTER_GL_OPTIONS,
        )
        self.main_scatter.setDepthValue(1)
        self.addItem(self.main_scatter)

        self.highlight_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=np.zeros((0, 4), dtype=np.float32),
            size=POINT_SIZE_PX,
            pxMode=True,
            glOptions=SCATTER_GL_OPTIONS,
        )
        self.highlight_scatter.setDepthValue(2)
        self.addItem(self.highlight_scatter)

        self._data_bounds: Optional[Tuple[float, float, float, float]] = None

    # -------- projection --------
    def projectionMatrix(self, region=None, viewport=None):
        if viewport is None:
            viewport = self.getViewport()
        _, _, vw, vh = viewport
        if vh <= 0 or vw <= 0:
            return QMatrix4x4()
        aspect = vw / vh
        hh = max(self.ortho_scale, 1e-6)
        hw = hh * aspect
        m = QMatrix4x4()
        m.ortho(-hw, hw, -hh, hh, -10000.0, 10000.0)
        return m

    # -------- 3D pos buffers --------
    def _project_2d(self, coords: np.ndarray) -> np.ndarray:
        if coords.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        d0, d1 = self.dims
        pos = np.zeros((coords.shape[0], 3), dtype=np.float32)
        pos[:, 0] = coords[:, d0]
        pos[:, 1] = coords[:, d1]
        return pos

    @staticmethod
    def _solid_color_array(n: int, r: float, g: float, b: float, a: float) -> np.ndarray:
        """Build an Nx4 colour array with one explicit value per point.
        Avoids the single-tuple-color codepath that v7 was hitting and
        which rendered everything white on some drivers."""
        if n == 0:
            return np.zeros((0, 4), dtype=np.float32)
        arr = np.empty((n, 4), dtype=np.float32)
        arr[:, 0] = r
        arr[:, 1] = g
        arr[:, 2] = b
        arr[:, 3] = a
        return arr

    def update_main(self, coords: np.ndarray):
        pos = self._project_2d(coords)
        col = self._solid_color_array(pos.shape[0], 0.0, 1.0, 0.0, 1.0)
        self.main_scatter.setData(pos=pos, color=col)

    def update_highlight(self, coords: np.ndarray):
        pos = self._project_2d(coords)
        col = self._solid_color_array(pos.shape[0], 1.0, 1.0, 0.0, 1.0)
        self.highlight_scatter.setData(pos=pos, color=col)

    def update_neighbors(
        self,
        pts: np.ndarray,
        ids: np.ndarray,
        color_for_id,
    ):
        """Build pos and per-point colour arrays from (pts, ids) and
        push them to the single neighbour scatter."""
        if pts.size == 0 or ids.size == 0:
            self.neighbor_scatter.setData(
                pos=np.zeros((0, 3), dtype=np.float32),
                color=np.zeros((0, 4), dtype=np.float32),
            )
            return

        d0, d1 = self.dims
        n = pts.shape[0]

        pos = np.zeros((n, 3), dtype=np.float32)
        pos[:, 0] = pts[:, d0]
        pos[:, 1] = pts[:, d1]

        col = np.empty((n, 4), dtype=np.float32)
        col[:, 3] = NEIGHBOR_ALPHA
        for nid in np.unique(ids):
            nid_int = int(nid)
            color_hex = color_for_id(nid_int)
            qc = QColor(color_hex)
            sel = (ids == nid)
            col[sel, 0] = qc.redF()
            col[sel, 1] = qc.greenF()
            col[sel, 2] = qc.blueF()

        self.neighbor_scatter.setData(pos=pos, color=col)

    def set_view_to_data(self, main_coords: np.ndarray, nbr_coords: np.ndarray):
        d0, d1 = self.dims
        bits = []
        if main_coords.size > 0:
            bits.append(main_coords[:, [d0, d1]])
        if nbr_coords.size > 0:
            bits.append(nbr_coords[:, [d0, d1]])
        if not bits:
            return
        all_pts = np.vstack(bits)
        xs = all_pts[:, 0]
        ys = all_pts[:, 1]
        cx = (float(xs.min()) + float(xs.max())) * 0.5
        cy = (float(ys.min()) + float(ys.max())) * 0.5
        range_w = float(xs.max()) - float(xs.min())
        range_h = float(ys.max()) - float(ys.min())
        self._data_bounds = (
            float(xs.min()), float(xs.max()),
            float(ys.min()), float(ys.max()),
        )
        w, h = self.width(), self.height()
        aspect = (w / h) if h > 0 else 1.0
        needed_h = (range_h * 0.5) * 1.05
        needed_w = (range_w * 0.5 / aspect) * 1.05 if aspect > 0 else needed_h
        self.ortho_scale = max(needed_h, needed_w, 1.0)
        self.opts['center'] = QVector3D(cx, cy, 0.0)
        self.update()

    # -------- mouse --------
    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._lasso_active = True
            self._lasso_pts = [(int(ev.pos().x()), int(ev.pos().y()))]
            ev.accept()
            self.update()
        elif ev.button() == Qt.RightButton:
            self._pan_active = True
            self._pan_last = (int(ev.pos().x()), int(ev.pos().y()))
            ev.accept()
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._lasso_active:
            self._lasso_pts.append((int(ev.pos().x()), int(ev.pos().y())))
            self.update()    # repaint to extend the yellow polyline
            ev.accept()
        elif self._pan_active:
            sx, sy = int(ev.pos().x()), int(ev.pos().y())
            dx_px = sx - self._pan_last[0]
            dy_px = sy - self._pan_last[1]
            w, h = self.width(), self.height()
            if w > 0 and h > 0:
                aspect = w / h
                wpp_x = (2.0 * self.ortho_scale * aspect) / w
                wpp_y = (2.0 * self.ortho_scale) / h
                c = self.opts['center']
            
                self.opts['center'] = QVector3D(
                    c.x() - dx_px * wpp_x,
                    c.y() + dy_px * wpp_y,
                    c.z(),
                )
                self._pan_last = (sx, sy)
                self.update()
            ev.accept()
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._lasso_active and ev.button() == Qt.LeftButton:
            screen_pts = list(self._lasso_pts)
            self._lasso_pts = []
            self._lasso_active = False
            self.update()    # repaint to clear the polyline
            if len(screen_pts) >= 3:
                world_pts = [self._screen_to_world(sx, sy) for sx, sy in screen_pts]
                self.sigLassoFinished.emit(world_pts, self.view_idx)
            ev.accept()
            return
        if self._pan_active and ev.button() == Qt.RightButton:
            self._pan_active = False
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def wheelEvent(self, ev):
        delta = ev.angleDelta().y()
        if delta == 0:
            return
        factor = (1.0 / 1.15) if delta > 0 else 1.15

        sx, sy = int(ev.pos().x()), int(ev.pos().y())
        wx_before, wy_before = self._screen_to_world(sx, sy)
        self.ortho_scale = max(self.ortho_scale * factor, 0.01)
        wx_after, wy_after = self._screen_to_world(sx, sy)
        c = self.opts['center']
        self.opts['center'] = QVector3D(
            c.x() - (wx_after - wx_before),
            c.y() - (wy_after - wy_before),
            c.z(),
        )

        self.update()
        ev.accept()

    # -------- screen <-> world --------
    def _screen_to_world(self, sx: int, sy: int) -> Tuple[float, float]:
        w, h = self.width(), self.height()
        if w == 0 or h == 0:
            return (0.0, 0.0)
        aspect = w / h
        ndc_x = (2.0 * sx / w) - 1.0
        ndc_y = 1.0 - (2.0 * sy / h)
        cam_x = ndc_x * self.ortho_scale * aspect
        cam_y = ndc_y * self.ortho_scale
        cx = self.opts['center'].x()
        cy = self.opts['center'].y()
        return (cam_x + cx, cam_y + cy)

    def _world_to_screen(self, wx: float, wy: float) -> Tuple[float, float]:
        w, h = self.width(), self.height()
        if w == 0 or h == 0:
            return (0.0, 0.0)
        aspect = w / h
        cx = self.opts['center'].x()
        cy = self.opts['center'].y()
        cam_x = wx - cx
        cam_y = wy - cy
        ndc_x = cam_x / (self.ortho_scale * aspect)
        ndc_y = cam_y / self.ortho_scale
        sx = (ndc_x + 1.0) * 0.5 * w
        sy = (1.0 - ndc_y) * 0.5 * h
        return (sx, sy)

    # -------- overlay (lasso + axes) --------
    def paintGL(self):
        super().paintGL()

        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            self._paint_axes(painter)
            self._paint_lasso(painter)
        finally:
            painter.end()

    def _paint_lasso(self, painter: QPainter):
        if not (self._lasso_active and len(self._lasso_pts) >= 2):
            return
        pen = QPen(QColor(255, 255, 0), 1.5)
        painter.setPen(pen)
        poly = QPolygon([QPoint(x, y) for x, y in self._lasso_pts])
        painter.drawPolyline(poly)

    def _paint_axes(self, painter: QPainter):
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return

        pen_axis = QPen(QColor(180, 180, 180), 1)
        pen_text = QPen(QColor(220, 220, 220), 1)
        painter.setPen(pen_axis)
        fm = QFontMetrics(painter.font())

        # Visible world ranges.
        wx0, wy0 = self._screen_to_world(0, h)
        wx1, wy1 = self._screen_to_world(w, 0)

        # --- bottom (x) axis ---
        y_bottom = h - 1
        painter.drawLine(0, y_bottom, w, y_bottom)
        x_ticks = _nice_ticks(wx0, wx1, target_count=6, integer_only=False)
        for tx in x_ticks:
            sx, _ = self._world_to_screen(tx, 0)
            sx_i = int(round(sx))
            painter.drawLine(sx_i, y_bottom, sx_i, y_bottom - TICK_LEN_PX)
            label = f"{tx:g}"
            tw = fm.horizontalAdvance(label)
            painter.setPen(pen_text)
            painter.drawText(sx_i - tw // 2, y_bottom - TICK_LEN_PX - 2, label)
            painter.setPen(pen_axis)

        # --- left (y) axis ---
        if self.show_y_labels:
            painter.drawLine(0, 0, 0, h)
            y_ticks = _nice_ticks(
                wy0, wy1, target_count=6, integer_only=self.integer_y,
            )
            for ty in y_ticks:
                _, sy = self._world_to_screen(0, ty)
                sy_i = int(round(sy))
                painter.drawLine(0, sy_i, TICK_LEN_PX, sy_i)
                if self.integer_y:
                    label = f"{int(round(ty))}"
                else:
                    label = f"{ty:g}"
                painter.setPen(pen_text)
                painter.drawText(TICK_LEN_PX + 2, sy_i + fm.height() // 3, label)
                painter.setPen(pen_axis)


# ============================================================
#  Main inspector class
# ============================================================
class TreeInspector:
    def __init__(self, tree_dir: str, out_dir: Optional[str] = None, id_field: str = "PredInstance"):
        self.tree_dir = tree_dir
        self.out_dir  = out_dir if out_dir is not None else tree_dir
        self.id_field = id_field
        self.next_new_tree_id: Optional[int] = None

        self.stop_flag:       bool = False
        self.save_requested:  bool = False

        self.current_pc:       Optional[np.ndarray] = None
        self.current_indices:  Optional[np.ndarray] = None
        self._original_tree_indices: Optional[np.ndarray] = None
        self._local_origin:    np.ndarray = np.zeros(3, dtype=float)

        self.neighbor_pc:  np.ndarray = np.empty((0, 3), dtype=float)
        self.neighbor_ids: np.ndarray = np.empty((0,),   dtype=int)

        self.history:    List[dict] = []
        self.max_history = 20

        self.reassignments:           List[PendingReassignment]    = []
        self.finalized_reassignments: List[Tuple[np.ndarray, int]] = []

        self.id_colors: Dict[int, str] = {}
        self._tree_bounds: Optional[Tuple[float, float, float, float]] = None

        self._dialog: Optional[QDialog] = None
        self._views:  List[OrthoGLView] = []
        self._dims:   List[Tuple[int, int]] = [(0, 1), (0, 2), (1, 2)]
        self._highlight_mask: Optional[np.ndarray] = None

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

            try:
                tree_ids = np.asarray(getattr(las, self.id_field)).copy()
            except AttributeError:
                print(f"  Field '{self.id_field}' not found. Skipping.")
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

                x_abs = x[mask_main]; y_abs = y[mask_main]; z_abs = z[mask_main]
                x_abs_min, x_abs_max = float(x_abs.min()), float(x_abs.max())
                y_abs_min, y_abs_max = float(y_abs.min()), float(y_abs.max())

                self._local_origin = np.array([
                    np.floor(x_abs_min), np.floor(y_abs_min), 0.0
                ], dtype=float)

                self.current_pc = np.column_stack((x_abs, y_abs, z_abs)).astype(float)
                self.current_pc -= self._local_origin
                self.current_indices = original_tree_indices.copy()

                x_min = float(self.current_pc[:, 0].min())
                x_max = float(self.current_pc[:, 0].max())
                y_min = float(self.current_pc[:, 1].min())
                y_max = float(self.current_pc[:, 1].max())
                self._tree_bounds = (x_min, x_max, y_min, y_max)

                m = NEIGHBOR_MARGIN
                mask_neighbors = (
                    (tree_ids != tid)
                    & (tree_ids != UNCLASSIFIED_ID)
                    & (x >= (x_abs_min - m)) & (x <= (x_abs_max + m))
                    & (y >= (y_abs_min - m)) & (y <= (y_abs_max + m))
                )

                if np.any(mask_neighbors):
                    self.neighbor_pc = np.column_stack(
                        (x[mask_neighbors], y[mask_neighbors], z[mask_neighbors])
                    ).astype(float)
                    self.neighbor_pc -= self._local_origin
                    self.neighbor_ids = tree_ids[mask_neighbors].astype(int, copy=False)
                else:
                    self.neighbor_pc  = np.empty((0, 3), dtype=float)
                    self.neighbor_ids = np.empty((0,),   dtype=int)

                self.reassignments           = []
                self.finalized_reassignments = []
                self.history                 = []
                self.save_requested          = False

                self.open_gui(f_path, int(tid))

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
        self._highlight_mask = np.zeros(len(self.current_pc), dtype=bool)

        dlg = QDialog()
        dlg.setAttribute(Qt.WA_DeleteOnClose, True)
        dlg.setWindowTitle(
            f"Tree ID {tid} | {os.path.basename(filename)} | field: {self.id_field} "
            f"| Left=Lasso  Right=Pan  Wheel=Zoom"
        )
        dlg.resize(1800, 800)
        self._dialog = dlg

        root = QVBoxLayout(dlg)

        grid = QGridLayout()
        grid.setSpacing(4)
        titles = ["XY Top View", "XZ Front View", "YZ Side View"]
        axis_x = ["X", "X", "Y"]
        axis_y = ["Y", "Z", "Z"]
        show_y     = [False, True,  True]
        integer_y  = [False, True,  True]

        self._views = []
        for i, (title, dims) in enumerate(zip(titles, self._dims)):
            cell = QVBoxLayout()
            cell.setContentsMargins(0, 0, 0, 0)
            cell.setSpacing(2)

            label = QLabel(title)
            label.setStyleSheet("color: #ddd; font-weight: bold;")
            label.setAlignment(Qt.AlignCenter)
            cell.addWidget(label)

            view = OrthoGLView(
                dims=dims,
                view_idx=i,
                show_y_labels=show_y[i],
                integer_y=integer_y[i],
                x_label=axis_x[i],
                y_label=axis_y[i],
            )
            view.sigLassoFinished.connect(self._on_lasso_finished)
            cell.addWidget(view, 1)

            wrapper = QWidget()
            wrapper.setLayout(cell)
            grid.addWidget(wrapper, 0, i)
            self._views.append(view)

        grid_wrap = QWidget()
        grid_wrap.setLayout(grid)
        root.addWidget(grid_wrap, 1)

        btn_row = QHBoxLayout()
        btn_undo  = QPushButton("Undo");          btn_undo.clicked.connect(self.undo)
        btn_apply = QPushButton("Apply && Stay")
        btn_apply.setStyleSheet("background-color: #3498db; color: white; font-weight: bold;")
        btn_apply.clicked.connect(self.apply_changes)
        btn_erase = QPushButton("Erase Tree");    btn_erase.clicked.connect(self.erase_tree)
        btn_save  = QPushButton("Save && Next")
        btn_save.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold;")
        btn_save.clicked.connect(self.save_exit)
        btn_skip  = QPushButton("Skip");          btn_skip.clicked.connect(self.skip_exit)
        btn_stop  = QPushButton("Exit");          btn_stop.clicked.connect(self.stop_all)
        for b in (btn_undo, btn_apply, btn_erase, btn_save, btn_skip, btn_stop):
            btn_row.addWidget(b)
        root.addLayout(btn_row)

        def _initial_frame():
            self._push_all_to_gl()
            for v in self._views:
                v.set_view_to_data(
                    self.current_pc if self.current_pc is not None else np.empty((0, 3)),
                    self.neighbor_pc,
                )
        QTimer.singleShot(0, _initial_frame)

        dlg.exec_()

        self._dialog = None
        self._views  = []

    # ----------------------------
    # Push to GL
    # ----------------------------
    def _push_all_to_gl(self):
        cur = self.current_pc if self.current_pc is not None else np.empty((0, 3))
        for v in self._views:
            v.update_main(cur)
            mask = self._visible_neighbor_mask_for_view(v.view_idx)
            pts = self.neighbor_pc[mask] if self.neighbor_pc.size > 0 else self.neighbor_pc
            ids = self.neighbor_ids[mask] if self.neighbor_ids.size > 0 else self.neighbor_ids
            v.update_neighbors(pts, ids, self._color_for_id)
            if self._highlight_mask is not None and self.current_pc is not None and np.any(self._highlight_mask):
                v.update_highlight(self.current_pc[self._highlight_mask])
            else:
                v.update_highlight(np.empty((0, 3)))

    def _push_highlight_to_gl(self):
        if self._highlight_mask is None or self.current_pc is None:
            for v in self._views:
                v.update_highlight(np.empty((0, 3)))
            return
        coords = self.current_pc[self._highlight_mask] if np.any(self._highlight_mask) else np.empty((0, 3))
        for v in self._views:
            v.update_highlight(coords)

    def _push_main_to_gl(self):
        cur = self.current_pc if self.current_pc is not None else np.empty((0, 3))
        for v in self._views:
            v.update_main(cur)

    def _push_neighbors_to_gl(self):
        for v in self._views:
            mask = self._visible_neighbor_mask_for_view(v.view_idx)
            pts = self.neighbor_pc[mask] if self.neighbor_pc.size > 0 else self.neighbor_pc
            ids = self.neighbor_ids[mask] if self.neighbor_ids.size > 0 else self.neighbor_ids
            v.update_neighbors(pts, ids, self._color_for_id)

    # ----------------------------
    # Visibility mask 
    # ----------------------------
    def _visible_neighbor_mask_for_view(self, view_index: int) -> np.ndarray:
        if self.neighbor_pc.size == 0 or self._tree_bounds is None:
            return np.zeros((self.neighbor_pc.shape[0],), dtype=bool)
        x_min, x_max, y_min, y_max = self._tree_bounds
        m = NEIGHBOR_MARGIN
        n = SIDE_MARGIN
        if view_index == 1:
            return (
                (self.neighbor_pc[:, 0] >= x_min - m) & (self.neighbor_pc[:, 0] <= x_max + m) &
                (self.neighbor_pc[:, 1] >= y_min - n) & (self.neighbor_pc[:, 1] <= y_max + n)
            )
        if view_index == 2:
            return (
                (self.neighbor_pc[:, 1] >= y_min - m) & (self.neighbor_pc[:, 1] <= y_max + m) &
                (self.neighbor_pc[:, 0] >= x_min - n) & (self.neighbor_pc[:, 0] <= x_max + n)
            )
        return np.ones((self.neighbor_pc.shape[0],), dtype=bool)

    # ----------------------------
    # Selection 
    # ----------------------------
    def _on_lasso_finished(self, verts, view_idx: int):
        if self.current_pc is None or self.current_pc.size == 0:
            return

        d0, d1 = self._dims[view_idx]

        arr = np.asarray(verts, dtype=float)
        if arr.shape[0] < 3:
            return

        lx0, lx1 = float(arr[:, 0].min()), float(arr[:, 0].max())
        ly0, ly1 = float(arr[:, 1].min()), float(arr[:, 1].max())

        xs = self.current_pc[:, d0]
        ys = self.current_pc[:, d1]
        in_bbox = (xs >= lx0) & (xs <= lx1) & (ys >= ly0) & (ys <= ly1)
        if not np.any(in_bbox):
            return

        cand_idx = np.flatnonzero(in_bbox)
        pts2d    = np.column_stack((xs[cand_idx], ys[cand_idx]))
        inside   = Path(arr).contains_points(pts2d)
        final_idx = cand_idx[inside]
        if final_idx.size == 0:
            return

        self._highlight_mask = np.zeros(len(self.current_pc), dtype=bool)
        self._highlight_mask[final_idx] = True
        self._push_highlight_to_gl()
        self._qapp.processEvents()

        target_id = self.ask_new_id()

        self._highlight_mask = np.zeros(len(self.current_pc), dtype=bool)
        self._push_highlight_to_gl()

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

        selected_indices = self.current_indices[final_idx].astype(int, copy=False)
        selected_coords  = self.current_pc[final_idx].copy()

        self.reassignments.append(PendingReassignment(
            indices=selected_indices.copy(),
            new_id=int(target_id),
            coords=selected_coords,
        ))

        keep = np.ones(len(self.current_pc), dtype=bool)
        keep[final_idx] = False
        self.current_pc      = self.current_pc[keep]
        self.current_indices = self.current_indices[keep]
        self._highlight_mask = np.zeros(len(self.current_pc), dtype=bool)

        self._push_main_to_gl()
        self._push_highlight_to_gl()

    # ----------------------------
    # Actions
    # ----------------------------
    def apply_changes(self, event=None):
        if not self.reassignments:
            return
        self._push_history({
            "current_pc":      self.current_pc.copy(),
            "current_indices": self.current_indices.copy(),
            "reassignments":   self._copy_reassignments(self.reassignments),
            "neighbor_pc":     self.neighbor_pc.copy(),
            "neighbor_ids":    self.neighbor_ids.copy(),
            "finalized":       list(self.finalized_reassignments),
        })
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
            self._push_neighbors_to_gl()
        except Exception as e:
            print("Exception in apply_changes:", e)
            traceback.print_exc()

    def undo(self, event=None):
        if not self.history:
            return
        state = self.history.pop()
        self.current_pc              = state["current_pc"]
        self.current_indices         = state["current_indices"]
        self.reassignments           = self._copy_reassignments(state["reassignments"])
        self.neighbor_pc             = state["neighbor_pc"]
        self.neighbor_ids            = state["neighbor_ids"]
        self.finalized_reassignments = list(state["finalized"])
        self._highlight_mask         = np.zeros(len(self.current_pc), dtype=bool)
        self._push_all_to_gl()

    def erase_tree(self, event=None):
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
        self._highlight_mask = np.zeros((0,), dtype=bool)
        self._push_main_to_gl()
        self._push_highlight_to_gl()

    def _push_history(self, state):
        self.history.append(state)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    # ----------------------------
    # Save / Skip / Exit
    # ----------------------------
    def save_exit(self, event=None):
        self.save_requested = True
        if self._dialog is not None:
            self._dialog.accept()

    def skip_exit(self, event=None):
        self.save_requested = False
        if self._dialog is not None:
            self._dialog.accept()

    def stop_all(self, event=None):
        self.stop_flag = True
        if self._dialog is not None:
            self._dialog.accept()

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
        layout = QVBoxLayout(); dlg.setLayout(layout)
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

        btn_new = QPushButton(f"+ New Tree (ID {self.next_new_tree_id})")
        btn_new.setStyleSheet("background-color: #ccffcc;")
        btn_new.clicked.connect(select_new_tree)
        layout.addWidget(btn_new)

        if not selectable_ids:
            layout.addWidget(QLabel("(No neighbor trees found nearby)"))
        else:
            for nid in selectable_ids:
                color = self.id_colors.get(nid, "#aaaaaa")
                row = QWidget()
                row_layout = QHBoxLayout(row); row_layout.setContentsMargins(0, 0, 0, 0)
                swatch = QFrame(); swatch.setFixedSize(15, 15)
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
        self.id_colors  = {}
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
            self.id_colors[n_id] = "#FFFFFF"; return
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

def main():
    qapp = QApplication.instance() or QApplication(sys.argv)

    dlg = LaunchDialog()
    if dlg.exec_() != QDialog.Accepted:
        print("Launch cancelled.")
        sys.exit(0)

    inspector = TreeInspector(
        tree_dir=dlg.input_dir,
        out_dir=dlg.output_dir,
        id_field=dlg.id_field,
    )
    inspector.run()


if __name__ == "__main__":
    main()


