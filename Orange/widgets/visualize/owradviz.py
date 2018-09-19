from itertools import islice, permutations, chain
from math import factorial
import warnings

import numpy as np
from scipy.spatial import distance
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from AnyQt.QtGui import QStandardItem, QColor
from AnyQt.QtCore import Qt, QRectF, QPoint, pyqtSignal as Signal
from AnyQt.QtWidgets import qApp, QApplication, QToolTip, QGraphicsEllipseItem

import pyqtgraph as pg
from pyqtgraph.graphicsItems.ScatterPlotItem import ScatterPlotItem

from Orange.data import Table, Domain, StringVariable
from Orange.preprocess.score import ReliefF, RReliefF
from Orange.projection import radviz
from Orange.widgets import widget, gui
from Orange.widgets.gui import OWComponent
from Orange.widgets.settings import Setting, ContextSetting, SettingProvider
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.plot import VariablesSelection
from Orange.widgets.visualize.utils.widget import OWProjectionWidget
from Orange.widgets.visualize.utils import VizRankDialog
from Orange.widgets.visualize.utils.component import OWVizGraph
from Orange.widgets.visualize.utils.plotutils import (
    TextItem, VizInteractiveViewBox
)
from Orange.widgets.widget import Output


class RadvizVizRank(VizRankDialog, OWComponent):
    captionTitle = "Score Plots"
    n_attrs = Setting(3)
    minK = 10

    attrsSelected = Signal([])
    _AttrRole = next(gui.OrangeUserRole)

    percent_data_used = Setting(100)

    def __init__(self, master):
        """Add the spin box for maximal number of attributes"""
        VizRankDialog.__init__(self, master)
        OWComponent.__init__(self, master)

        self.master = master
        self.n_neighbors = 10
        max_n_attrs = len(master.model_selected) + len(master.model_other) - 1

        box = gui.hBox(self)
        self.n_attrs_spin = gui.spin(
            box, self, "n_attrs", 3, max_n_attrs, label="Maximum number of variables: ",
            controlWidth=50, alignment=Qt.AlignRight, callback=self._n_attrs_changed)
        gui.rubber(box)
        self.last_run_n_attrs = None
        self.attr_color = master.attr_color
        self.attr_ordering = None
        self.data = None
        self.valid_data = None

    def initialize(self):
        super().initialize()
        self.attr_color = self.master.attr_color

    def _compute_attr_order(self):
        """
        used by VizRank to evaluate attributes
        """
        master = self.master
        attrs = [v for v in chain(master.model_selected[:], master.model_other[:])
                 if v is not self.attr_color]
        data = self.master.data.transform(Domain(attributes=attrs, class_vars=self.attr_color))
        self.data = data
        self.valid_data = np.hstack((~np.isnan(data.X), ~np.isnan(data.Y.reshape(len(data.Y), 1))))
        relief = ReliefF if self.attr_color.is_discrete else RReliefF
        weights = relief(n_iterations=100, k_nearest=self.minK)(data)
        attrs = sorted(zip(weights, attrs), key=lambda x: (-x[0], x[1].name))
        self.attr_ordering = attr_ordering = [a for _, a in attrs]
        return attr_ordering

    def _evaluate_projection(self, x, y):
        """
        kNNEvaluate - evaluate class separation in the given projection using a k-NN method
        Parameters
        ----------
        x - variables to evaluate
        y - class

        Returns
        -------
        scores
        """
        if self.percent_data_used != 100:
            rand = np.random.choice(len(x), int(len(x) * self.percent_data_used / 100),
                                    replace=False)
            x = x[rand]
            y = y[rand]
        neigh = KNeighborsClassifier(n_neighbors=3) if self.attr_color.is_discrete else \
            KNeighborsRegressor(n_neighbors=3)
        assert ~(np.isnan(x).any(axis=None) | np.isnan(x).any(axis=None))
        neigh.fit(x, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            scores = cross_val_score(neigh, x, y, cv=3)
        return scores.mean()

    def _n_attrs_changed(self):
        """
        Change the button label when the number of attributes changes. The method does not reset
        anything so the user can still see the results until actually restarting the search.
        """
        if self.n_attrs != self.last_run_n_attrs or self.saved_state is None:
            self.button.setText("Start")
        else:
            self.button.setText("Continue")
        self.button.setEnabled(self.check_preconditions())

    def progressBarSet(self, value, processEvents=None):
        self.setWindowTitle(self.captionTitle + " Evaluated {} permutations".format(value))
        if processEvents is not None and processEvents is not False:
            qApp.processEvents(processEvents)

    def check_preconditions(self):
        master = self.master
        if not super().check_preconditions():
            return False
        elif not master.btn_vizrank.isEnabled():
            return False
        self.n_attrs_spin.setMaximum(20)  # all primitive vars except color one
        return True

    def on_selection_changed(self, selected, deselected):
        attrs = selected.indexes()[0].data(self._AttrRole)
        self.selectionChanged.emit([attrs])

    def iterate_states(self, state):
        if state is None:  # on the first call, compute order
            self.attrs = self._compute_attr_order()
            state = list(range(3))
        else:
            state = list(state)

        def combinations(n, s):
            while True:
                yield s
                for up, _ in enumerate(s):
                    s[up] += 1
                    if up + 1 == len(s) or s[up] < s[up + 1]:
                        break
                    s[up] = up
                if s[-1] == n:
                    if len(s) < self.n_attrs:
                        s = list(range(len(s) + 1))
                    else:
                        break

        for c in combinations(len(self.attrs), state):
            for p in islice(permutations(c[1:]), factorial(len(c) - 1) // 2):
                yield (c[0],) + p

    def compute_score(self, state):
        attrs = [self.attrs[i] for i in state]
        domain = Domain(attributes=attrs, class_vars=[self.attr_color])
        data = self.data.transform(domain)
        radviz_xy, _, mask = radviz(data, attrs)
        y = data.Y[mask]
        return -self._evaluate_projection(radviz_xy, y)

    def bar_length(self, score):
        return -score

    def row_for_state(self, score, state):
        attrs = [self.attrs[s] for s in state]
        item = QStandardItem("[{:0.6f}] ".format(-score) + ", ".join(a.name for a in attrs))
        item.setData(attrs, self._AttrRole)
        return [item]

    def _update_progress(self):
        self.progressBarSet(int(self.saved_progress))

    def before_running(self):
        """
        Disable the spin for number of attributes before running and
        enable afterwards. Also, if the number of attributes is different than
        in the last run, reset the saved state (if it was paused).
        """
        if self.n_attrs != self.last_run_n_attrs:
            self.saved_state = None
            self.saved_progress = 0
        if self.saved_state is None:
            self.scores = []
            self.rank_model.clear()
        self.last_run_n_attrs = self.n_attrs
        self.n_attrs_spin.setDisabled(True)

    def stopped(self):
        self.n_attrs_spin.setDisabled(False)


class RadvizInteractiveViewBox(VizInteractiveViewBox):
    def mouseDragEvent(self, ev, axis=None):
        super().mouseDragEvent(ev, axis)
        if ev.finish:
            self.setCursor(Qt.ArrowCursor)
            self.graph.show_indicator(None)

    def _show_tooltip(self, ev):
        pos = self.childGroup.mapFromParent(ev.pos())
        angle = np.arctan2(pos.y(), pos.x())
        point = QPoint(ev.screenPos().x(), ev.screenPos().y())
        QToolTip.showText(point, "{:.2f}".format(np.rad2deg(angle)))


class OWRadvizGraph(OWVizGraph):
    def __init__(self, scatter_widget, parent):
        super().__init__(scatter_widget, parent, RadvizInteractiveViewBox)
        self._text_items = []

    def set_point(self, i, x, y):
        angle = np.arctan2(y, x)
        super().set_point(i, np.cos(angle), np.sin(angle))

    def set_view_box_range(self):
        self.view_box.setRange(RANGE, padding=0.025)

    def can_show_indicator(self, pos):
        if self._points is None:
            return False, None

        np_pos = np.array([[pos.x(), pos.y()]])
        distances = distance.cdist(np_pos, self._points[:, :2])[0]
        if len(distances) and np.min(distances) < self.DISTANCE_DIFF:
            return True, np.argmin(distances)
        return False, None

    def update_items(self):
        super().update_items()
        self._update_text_items()

    def _update_text_items(self):
        self._remove_text_items()
        self._add_text_items()

    def _remove_text_items(self):
        for item in self._text_items:
            self.plot_widget.removeItem(item)
        self._text_items = []

    def _add_text_items(self):
        if self._points is None:
            return
        for point in self._points:
            ti = TextItem()
            ti.setText(point[2].name)
            ti.setColor(QColor(0, 0, 0))
            ti.setPos(point[0], point[1])
            self._text_items.append(ti)
            self.plot_widget.addItem(ti)

    def _add_point_items(self):
        if self._points is None:
            return
        x, y = self._points[:, 0], self._points[:, 1]
        self._point_items = ScatterPlotItem(x=x, y=y)
        self.plot_widget.addItem(self._point_items)

    def _add_circle_item(self):
        if self._points is None:
            return
        self._circle_item = QGraphicsEllipseItem()
        self._circle_item.setRect(QRectF(-1., -1., 2., 2.))
        self._circle_item.setPen(pg.mkPen(QColor(0, 0, 0), width=2))
        self.plot_widget.addItem(self._circle_item)

    def _add_indicator_item(self, point_i):
        if point_i is None:
            return
        x, y = self._points[point_i][:2]
        col = self.view_box.mouse_state
        dx = (self.view_box.childGroup.mapToDevice(QPoint(1, 0)) -
              self.view_box.childGroup.mapToDevice(QPoint(-1, 0))).x()
        self._indicator_item = MoveIndicator(np.arctan2(y, x), col, 6000 / dx)
        self.plot_widget.addItem(self._indicator_item)


RANGE = QRectF(-1.2, -1.05, 2.4, 2.1)
MAX_POINTS = 100


class OWRadviz(OWProjectionWidget):
    name = "Radviz"
    description = "Display Radviz projection"
    icon = "icons/Radviz.svg"
    priority = 241
    keywords = ["viz"]

    class Outputs(OWProjectionWidget.Outputs):
        components = Output("Components", Table)

    settings_version = 2

    selected_vars = ContextSetting([])
    vizrank = SettingProvider(RadvizVizRank)
    GRAPH_CLASS = OWRadvizGraph
    graph = SettingProvider(OWRadvizGraph)
    embedding_variables_names = ("radviz-x", "radviz-y")

    class Warning(OWProjectionWidget.Warning):
        no_features = widget.Msg("At least 2 features have to be chosen")
        invalid_embedding = widget.Msg("No projection for selected features")

    class Error(OWProjectionWidget.Error):
        no_features = widget.Msg(
            "At least 3 numeric or categorical variables are required"
        )
        no_instances = widget.Msg("At least 2 data instances are required")

    def __init__(self):
        self.model_selected = VariableListModel(enable_dnd=True)
        self.model_selected.rowsInserted.connect(self.__model_selected_changed)
        self.model_selected.rowsRemoved.connect(self.__model_selected_changed)
        self.model_other = VariableListModel(enable_dnd=True)

        self.vizrank, self.btn_vizrank = RadvizVizRank.add_vizrank(
            None, self, "Suggest features", self.vizrank_set_attrs
        )
        super().__init__()
        self._rand_indices = None

        self.graph.view_box.started.connect(self._randomize_indices)
        self.graph.view_box.moved.connect(self._manual_move)
        self.graph.view_box.finished.connect(self._finish_manual_move)

    def _add_top_controls(self):
        self.variables_selection = VariablesSelection(
            self, self.model_selected, self.model_other, self.controlArea
        )
        self.variables_selection.add_remove.layout().addWidget(
            self.btn_vizrank
        )
        super()._add_top_controls()

    def _add_bottom_controls(self):
        self.graph.box_zoom_select(self.controlArea)
        gui.auto_commit(self.controlArea, self, "auto_commit",
                        "Send Selection", "Send Automatically")

    def vizrank_set_attrs(self, attrs):
        if not attrs:
            return
        self.variables_selection.display_none()
        self.model_selected[:] = attrs[:]
        self.model_other[:] = [v for v in self.model_other if v not in attrs]

    def update_colors(self):
        self._vizrank_color_change()
        self.cb_class_density.setEnabled(self.can_draw_density())

    def __model_selected_changed(self):
        self.selected_vars = [var.name for var in self.model_selected]
        self.init_embedding_coords()
        self.setup_plot()
        self.commit()

    def _vizrank_color_change(self):
        is_enabled = self.data is not None and not self.data.is_sparse() and \
            len(self.model_other) + len(self.model_selected) > 3 and \
            len(self.data[self.valid_data]) > 1 and \
            np.all(np.nan_to_num(np.nanstd(self.data.X, 0)) != 0)
        self.btn_vizrank.setEnabled(
            is_enabled and self.attr_color is not None
            and not np.isnan(self.data.get_column_view(
                self.attr_color)[0].astype(float)).all())
        self.vizrank.initialize()

    def clear(self):
        super().clear()
        self._rand_indices = None
        self.model_selected.clear()
        self.model_other.clear()
        self.graph.set_attributes(())
        self.graph.set_points(None)

    def set_data(self, data):
        super().set_data(data)
        if self.data is not None and self.data.domain is not None and \
                all([name in self.data.domain for name
                     in self.selected_vars]) and len(self.selected_vars):
            d, selected = self.data.domain, self.selected_vars
            variables = d.attributes + d.metas + d.class_vars
            variables = [v for v in variables if v.is_primitive()]
            self.model_selected[:] = [d[name] for name in selected]
            self.model_other[:] = [self.data.domain[attr.name] for attr in
                                   variables if attr.name not in selected]
        self._vizrank_color_change()

    def check_data(self):
        super().check_data()
        if self.data is not None:
            domain = self.data.domain
            if len(self.data) < 2:
                self.Error.no_instances()
                self.data = None
            elif len([v for v in domain.variables + domain.metas
                      if v.is_primitive()]) < 3:
                self.Error.no_features()
                self.data = None
            elif np.all(np.isnan(self.data.X).any(axis=1)):
                self.Warning.missing_coords()
                self.data = None

    def init_attr_values(self):
        super().init_attr_values()
        if self.data is not None:
            domain = self.data.domain
            variables = [v for v in domain.attributes + domain.metas
                         if v.is_primitive()]
            self.model_selected[:] = variables[:5]
            self.model_other[:] = variables[5:] + list(domain.class_vars)

    def init_embedding_coords(self):
        self.clear_messages()
        if len(self.model_selected) < 2:
            self.Warning.no_features()
            self.graph.clear()
            self._embedding_coords = None
            return

        r = radviz(self.data, self.model_selected)
        self._embedding_coords = r[0]
        self.graph.set_points(r[1])
        self.valid_data = r[2]

        if self._embedding_coords is None or \
                np.any(np.isnan(self._embedding_coords)):
            self.Warning.invalid_embedding()
            self._embedding_coords = None

    def setup_plot(self):
        if self._embedding_coords is not None:
            self.graph.reset_graph()

    def _randomize_indices(self):
        n = len(self._embedding_coords)
        if n > MAX_POINTS:
            self._rand_indices = np.random.choice(n, MAX_POINTS, replace=False)
            self._rand_indices = sorted(self._rand_indices)

    def _manual_move(self):
        res = radviz(self.data, self.model_selected, self.graph.get_points())
        self._embedding_coords = res[0]
        if self._rand_indices is not None:
            # save widget state
            selection = self.graph.selection
            valid_data = self.valid_data.copy()
            data = self.data.copy()
            ec = self._embedding_coords.copy()

            # plot subset
            self.__plot_random_subset(selection)

            # restore widget state
            self.graph.selection = selection
            self.valid_data = valid_data
            self.data = data
            self._embedding_coords = ec
        else:
            self.graph.update_coordinates()

    def __plot_random_subset(self, selection):
        self._embedding_coords = self._embedding_coords[self._rand_indices]
        self.data = self.data[self._rand_indices]
        self.valid_data = self.valid_data[self._rand_indices]
        self.graph.reset_graph()
        if selection is not None:
            self.graph.selection = selection[self._rand_indices]
            self.graph.update_selection_colors()

    def _finish_manual_move(self):
        if self._rand_indices is not None:
            selection = self.graph.selection
            self.graph.reset_graph()
            if selection is not None:
                self.graph.selection = selection
                self.graph.select_by_index(self.graph.get_selection())
        self.commit()

    def commit(self):
        super().commit()
        self.send_components()

    def send_components(self):
        components = None
        if self.data is not None and self.valid_data is not None and \
                self._embedding_coords is not None:
            points = self.graph.get_points()
            angle = np.arctan2(np.array(points[:, 1].T, dtype=float),
                               np.array(points[:, 0].T, dtype=float))
            meta_attrs = [StringVariable(name='component')]
            domain = Domain(points[:, 2], metas=meta_attrs)
            components = Table(domain, np.row_stack((points[:, :2].T, angle)),
                               metas=np.array([["RX"], ["RY"], ["angle"]]))
            components.name = self.data.name
        self.Outputs.components.send(components)

    @classmethod
    def migrate_context(cls, context, version):
        if version < 2:
            values = context.values
            values["attr_color"] = values["graph"]["attr_color"]
            values["attr_size"] = values["graph"]["attr_size"]
            values["attr_shape"] = values["graph"]["attr_shape"]
            values["attr_label"] = values["graph"]["attr_label"]


class MoveIndicator(pg.GraphicsObject):
    def __init__(self, angle, col, dangle=5, parent=None):
        super().__init__(parent)
        color = QColor(0, 0, 0) if col else QColor(128, 128, 128)
        angle_d = np.rad2deg(angle)
        angle_2 = 90 - angle_d - dangle
        angle_1 = 270 - angle_d + dangle
        dangle = np.deg2rad(dangle)
        arrow1 = pg.ArrowItem(
            parent=self, angle=angle_1, brush=color, pen=pg.mkPen(color)
        )
        arrow1.setPos(np.cos(angle - dangle), np.sin(angle - dangle))
        arrow2 = pg.ArrowItem(
            parent=self, angle=angle_2, brush=color, pen=pg.mkPen(color)
        )
        arrow2.setPos(np.cos(angle + dangle), np.sin(angle + dangle))
        arc_x = np.fromfunction(
            lambda i: np.cos((angle - dangle) + (2 * dangle) * i / 120.),
            (121,), dtype=int
        )
        arc_y = np.fromfunction(
            lambda i: np.sin((angle - dangle) + (2 * dangle) * i / 120.),
            (121,), dtype=int
        )
        pg.PlotCurveItem(
            parent=self, x=arc_x, y=arc_y, pen=pg.mkPen(color), antialias=False
        )

    def paint(self, painter, option, widget):
        pass

    def boundingRect(self):
        return QRectF()


def main(argv=None):
    import sys

    argv = sys.argv[1:] if argv is None else argv
    if argv:
        filename = argv[0]
    else:
        filename = "heart_disease"

    data = Table(filename)

    app = QApplication([])
    w = OWRadviz()
    w.set_data(data)
    w.set_subset_data(data[::10])
    w.handleNewSignals()
    w.show()
    app.exec()
    w.saveSettings()


if __name__ == "__main__":
    import sys
    sys.exit(main())
