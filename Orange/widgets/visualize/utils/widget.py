from collections import Counter, defaultdict
from xml.sax.saxutils import escape
from math import log2

import numpy as np
from scipy import sparse as sp

from AnyQt.QtCore import QSize
from AnyQt.QtWidgets import QApplication

from Orange.data import Table, ContinuousVariable, Domain, Variable
from Orange.statistics.util import bincount

from Orange.widgets import gui, report
from Orange.widgets.settings import (
    Setting, ContextSetting, DomainContextHandler, SettingProvider
)
from Orange.widgets.utils.annotated_data import (
    create_annotated_table, ANNOTATED_DATA_SIGNAL_NAME, create_groups_table,
    get_unique_names
)
from Orange.widgets.utils.colorpalette import (
    ColorPaletteGenerator, ContinuousPaletteGenerator, DefaultRGBColors
)
from Orange.widgets.utils.plot import OWPlotGUI
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotBase
from Orange.widgets.widget import OWWidget, Input, Output, Msg


MAX_CATEGORIES = 11  # maximum number of colors or shapes (including Other)
MAX_POINTS_IN_TOOLTIP = 5


class OWProjectionWidgetBase(OWWidget):
    """
    Base widget for widgets that use attribute data to set the colors, labels,
    shapes and sizes of points.

    The widgets defines settings `attr_color`, `attr_label`, `attr_shape`
    and `attr_size`, but leaves defining the gui to the derived widgets.
    These are expected to have controls that manipulate these settings,
    and the controls are expected to use attribute models.

    The widgets also defines attributes `data` and `valid_data` and expects
    the derived widgets to use them to store an instances of `data.Table`
    and a bool `np.ndarray` with indicators of valid (that is, shown)
    data points.
    """
    attr_color = ContextSetting(None, required=ContextSetting.OPTIONAL)
    attr_label = ContextSetting(None, required=ContextSetting.OPTIONAL)
    attr_shape = ContextSetting(None, required=ContextSetting.OPTIONAL)
    attr_size = ContextSetting(None, required=ContextSetting.OPTIONAL)

    class Information(OWWidget.Information):
        missing_size = Msg(
            "Points with undefined '{}' are shown in smaller size")
        missing_shape = Msg(
            "Points with undefined '{}' are shown as crossed circles")

    def __init__(self):
        super().__init__()
        self.data = None
        self.valid_data = None

        self.set_palette()

    def init_attr_values(self):
        """
        Set the models for `attr_color`, `attr_shape`, `attr_size` and
        `attr_label`. All values are set to `None`, except `attr_color`
        which is set to the class variable if it exists.
        """
        data = self.data
        domain = data.domain if data and len(data) else None
        for attr in ("attr_color", "attr_shape", "attr_size", "attr_label"):
            getattr(self.controls, attr).model().set_domain(domain)
            setattr(self, attr, None)
        if domain is not None:
            self.attr_color = domain.class_var

    def get_coordinates_data(self):
        """A get coordinated method that returns no coordinates.

        Derived classes must override this method.
        """
        return None, None

    def get_subset_mask(self):
        """
        Return the bool array indicating the points in the subset

        The base method does nothing and would usually be overridden by
        a method that returns indicators from the subset signal.

        Do not confuse the subset with selection.

        Returns:
            (np.ndarray or `None`): a bool array of indicators
        """
        return None

    @staticmethod
    def __get_overlap_groups(x, y):
        coord_to_id = defaultdict(list)
        for i, xy in enumerate(zip(x, y)):
            coord_to_id[xy].append(i)
        return coord_to_id

    def get_column(self, attr, filter_valid=True,
                   merge_infrequent=False, return_labels=False):
        """
        Retrieve the data from the given column in the data table

        The method:
        - densifies sparse data,
        - converts arrays with dtype object to floats if the attribute is
          actually primitive,
        - filters out invalid data (if `filter_valid` is `True`),
        - merges infrequent (discrete) values into a single value
          (if `merge_infrequent` is `True`).

        Tha latter feature is used for shapes and labels, where only a
        set number (`MAX`) of different values is shown, and others are
        merged into category 'Other'. In this case, the method may return
        either the data (e.g. color indices, shape indices) or the list
        of retained values, followed by `['Other']`.

        Args:
            attr (:obj:~Orange.data.Variable): the column to extract
            filter_valid (bool): filter out invalid data (default: `True`)
            merge_infrequent (bool): merge infrequent values (default: `False`)
            return_labels (bool): return a list of labels instead of data
                (default: `False`)

        Returns:
            (np.ndarray): (valid) data from the column, or a list of labels
        """
        if attr is None:
            return None
        all_data = self.data.get_column_view(attr)[0]
        if sp.issparse(all_data):
            all_data = all_data.toDense()  # TODO -- just guessing; fix this!
        elif all_data.dtype == object and attr.is_primitive():
            all_data = all_data.astype(float)
        if filter_valid and self.valid_data is not None:
            all_data = all_data[self.valid_data]
        if not merge_infrequent or attr.is_continuous \
                or len(attr.values) <= MAX_CATEGORIES:
            return attr.values if return_labels else all_data
        dist = bincount(all_data, max_val=len(attr.values) - 1)
        infrequent = np.zeros(len(attr.values), dtype=bool)
        infrequent[np.argsort(dist[0])[:-(MAX_CATEGORIES-1)]] = True
        # If discrete variable has more than maximium allowed values,
        # less used values are joined as "Other"
        if return_labels:
            return [value for value, infreq in zip(attr.values, infrequent)
                    if not infreq] + ["Other"]
        else:
            result = all_data.copy()
            freq_vals = [i for i, f in enumerate(infrequent) if not f]
            for i, f in enumerate(infrequent):
                result[all_data == i] = MAX_CATEGORIES - 1 if f \
                    else freq_vals.index(i)
            return result

    # Sizes
    def get_size_data(self):
        """Return the column corresponding to `attr_size`"""
        if self.attr_size == OWPlotGUI.SizeByOverlap:
            x, y = self.get_coordinates_data()
            coord_to_id = self.__get_overlap_groups(x, y)
            overlaps = [len(coord_to_id[xy]) for xy in zip(x, y)]
            return [1 + log2(o) for o in overlaps]
        return self.get_column(self.attr_size)

    def impute_sizes(self, size_data):
        """
        Default imputation for size data

        Missing values are replaced by `MinShapeSize - 2`. Imputation is
        done in place.

        Args:
            size_data (np.ndarray): scaled points sizes
        """
        nans = np.isnan(size_data)
        if np.any(nans):
            size_data[nans] = self.graph.MinShapeSize - 2
            self.Information.missing_size(self.attr_size)
        else:
            self.Information.missing_size.clear()

    def sizes_changed(self):
        self.graph.update_sizes()
        self.graph.update_colors()

    # Colors
    def get_color_data(self):
        """Return the column corresponding to color data"""
        colors = self.get_column(self.attr_color, merge_infrequent=True)
        if self.attr_size == OWPlotGUI.SizeByOverlap:
            # color overlapping points by most frequent color
            x, y = self.get_coordinates_data()
            coord_to_id = self.__get_overlap_groups(x, y)
            majority_colors = np.empty(len(x))
            for i, xy in enumerate(zip(x, y)):
                cnt = Counter(colors[j] for j in coord_to_id[xy])
                majority_colors[i] = cnt.most_common(1)[0][0]
            return majority_colors
        return colors

    def get_color_labels(self):
        """
        Return labels for the color legend

        Returns:
            (list of str): labels
        """
        return self.get_column(self.attr_color, merge_infrequent=True,
                               return_labels=True)

    def is_continuous_color(self):
        """
        Tells whether the color is continuous

        Returns:
            (bool):
        """
        return self.attr_color is not None and self.attr_color.is_continuous

    def set_palette(self):
        """
        Set the graph palette suitable for the current `attr_color`

        This method is invoked by the plot's `get_data` and must be overridden
        if the widget offers coloring that is not based on attribute values.
        """
        if self.attr_color is None:
            self.graph.palette = None
            return
        colors = self.attr_color.colors
        if self.attr_color.is_discrete:
            self.graph.palette = ColorPaletteGenerator(
                number_of_colors=min(len(colors), MAX_CATEGORIES),
                rgb_colors=colors if len(colors) <= MAX_CATEGORIES
                else DefaultRGBColors)
        else:
            self.graph.palette = ContinuousPaletteGenerator(*colors)

    def can_draw_density(self):
        """
        Tells whether the current data and settings are suitable for drawing
        densities

        Returns:
            (bool):
        """
        return self.data is not None and self.data.domain is not None and \
            len(self.data) > 1 and self.attr_color is not None

    def colors_changed(self):
        self.graph.update_colors()

    # Labels
    def get_label_data(self, formatter=None):
        """Return the column corresponding to label data"""
        if self.attr_label:
            label_data = self.get_column(self.attr_label)
            return map(formatter or self.attr_label.str_val, label_data)
        return None

    def labels_changed(self):
        self.graph.update_labels()

    # Shapes
    def get_shape_data(self):
        """
        Return labels for the shape legend

        Returns:
            (list of str): labels
        """
        return self.get_column(self.attr_shape, merge_infrequent=True)

    def get_shape_labels(self):
        return self.get_column(self.attr_shape, merge_infrequent=True,
                               return_labels=True)

    def impute_shapes(self, shape_data, default_symbol):
        """
        Default imputation for shape data

        Missing values are replaced by `default_symbol`. Imputation is
        done in place.

        Args:
            shape_data (np.ndarray): scaled points sizes
            default_symbol (str): a string representing the symbol
        """
        if shape_data is None:
            return 0
        nans = np.isnan(shape_data)
        if np.any(nans):
            shape_data[nans] = default_symbol
            self.Information.missing_shape(self.attr_shape)
        else:
            self.Information.missing_shape.clear()
        return shape_data

    def shapes_changed(self):
        self.graph.update_shapes()

    # Tooltip
    def _point_tooltip(self, point_id, skip_attrs=()):
        def show_part(_point_data, singular, plural, max_shown, _vars):
            cols = [escape('{} = {}'.format(var.name, _point_data[var]))
                    for var in _vars[:max_shown + 2]
                    if _vars == domain.class_vars
                    or var not in skip_attrs][:max_shown]
            if not cols:
                return ""
            n_vars = len(_vars)
            if n_vars > max_shown:
                cols[-1] = "... and {} others".format(n_vars - max_shown + 1)
            return \
                "<b>{}</b>:<br/>".format(singular if n_vars < 2 else plural) \
                + "<br/>".join(cols)

        domain = self.data.domain
        parts = (("Class", "Classes", 4, domain.class_vars),
                 ("Meta", "Metas", 4, domain.metas),
                 ("Feature", "Features", 10, domain.attributes))

        point_data = self.data[point_id]
        return "<br/>".join(show_part(point_data, *columns)
                            for columns in parts)

    def get_tooltip(self, point_ids):
        """
        Return the tooltip string for the given points

        The method is called by the plot on mouse hover

        Args:
            point_ids (list): indices into `data`

        Returns:
            (str):
        """
        text = "<hr/>".join(self._point_tooltip(point_id)
                            for point_id in point_ids[:MAX_POINTS_IN_TOOLTIP])
        if len(point_ids) > MAX_POINTS_IN_TOOLTIP:
            text = "{} instances<hr/>{}<hr/>...".format(len(point_ids), text)
        return text

    def keyPressEvent(self, event):
        """Update the tip about using the modifier keys when selecting"""
        super().keyPressEvent(event)
        self.graph.update_tooltip(event.modifiers())

    def keyReleaseEvent(self, event):
        """Update the tip about using the modifier keys when selecting"""
        super().keyReleaseEvent(event)
        self.graph.update_tooltip(event.modifiers())

    # Legend
    def combined_legend(self):
        """Tells whether the shape and color legends are combined into one"""
        return self.attr_shape == self.attr_color


class OWProjectionWidget(OWProjectionWidgetBase):
    """Base class for projection widgets."""
    class Inputs:
        data = Input("Data", Table, default=True)
        data_subset = Input("Data Subset", Table)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    class Warning(OWProjectionWidgetBase.Warning):
        missing_coords = Msg(
            "Plot cannot be displayed because data"
            "is missing for all points")

    class Error(OWProjectionWidgetBase.Error):
        sparse_data = Msg("Sparse data is not supported")

    settingsHandler = DomainContextHandler()
    selection = Setting(None, schema_only=True)
    auto_commit = Setting(True)

    GRAPH_CLASS = OWScatterPlotBase
    graph = SettingProvider(OWScatterPlotBase)
    graph_name = "graph.plot_widget.plotItem"
    embedding_variables_names = ("proj-x", "proj-y")

    def __init__(self):
        super().__init__()
        self.subset_data = None
        self.subset_indices = None
        self._embedding_coords = None  # embeddings for valid data
        self.__pending_selection = self.selection

        # GUI
        self._effects_box = None
        self._plot_box = None
        self.setup_gui()

    @property
    def embedding_valid_data(self):  # embedding for valid data
        return self._embedding_coords

    @property
    def embedding_all_data(self):  # embedding for valid and invalid data
        if self.data is None:
            return None
        embedding = np.zeros((len(self.data), 2), dtype=np.float)
        embedding[self.valid_data] = self.embedding_valid_data
        return embedding

    # GUI
    def setup_gui(self):
        self._add_graph()
        self._add_top_controls()
        self._add_middle_controls()
        self._add_bottom_controls()

    def _add_graph(self):
        box = gui.vBox(self.mainArea, True, margin=0)
        self.graph = self.GRAPH_CLASS(self, box)
        box.layout().addWidget(self.graph.plot_widget)

    def _add_top_controls(self):
        self.graph.gui.point_properties_box(self.controlArea)

    def _add_middle_controls(self):
        self._effects_box = self.graph.gui.effects_box(self.controlArea)
        self._plot_box = self.graph.gui.plot_properties_box(self.controlArea)

    def _add_bottom_controls(self):
        self.controlArea.layout().addStretch(100)
        self.graph.box_zoom_select(self.controlArea)
        gui.auto_commit(self.controlArea, self, "auto_commit",
                        "Send Selection", "Send Automatically")

    # Input
    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        self.closeContext()
        self.clear()
        self.data = data
        self.check_data()
        self.init_attr_values()
        self.openContext(self.data)
        self.cb_class_density.setEnabled(self.can_draw_density())

    def check_data(self):
        """Handle error messages and setting self.valid_data property.
        Should be overridden most of the times when subclassed."""
        self.clear_messages()
        if self.data is not None:
            if self.data.is_sparse():
                self.Error.sparse_data()
                self.data = None
        self.valid_data = np.any(np.isfinite(self.data.X), 1) \
            if self.data is not None else None
        if self.data is not None and not len(self.data[self.valid_data]):
            self.data = None
            self.Warning.missing_coords()

    @Inputs.data_subset
    @check_sql_input
    def set_subset_data(self, subset):
        self.subset_data = subset
        self.subset_indices = {e.id for e in subset} \
            if subset is not None else {}
        self.controls.graph.alpha_value.setEnabled(subset is None)

    def handleNewSignals(self):
        if self.data is not None:
            self.init_embedding_coords()
            self.setup_plot()
            self.apply_selection()
        self.commit()

    def get_subset_mask(self):
        if self.subset_indices:
            return np.array([ex.id in self.subset_indices
                             for ex in self.data[self.valid_data]])
        return None

    # Plot
    def init_embedding_coords(self):
        """Setting self._embedding_coords.
        Should be overridden when subclassed."""
        x_data = self.data.X
        x_data[x_data == np.inf] = np.nan
        x_data = np.nanmean(x_data[self.valid_data], 1)
        y_data = np.ones(len(x_data))
        self._embedding_coords = np.vstack((x_data, y_data)).T

    def get_coordinates_data(self):
        return (self._embedding_coords[:, 0], self._embedding_coords[:, 1]) \
            if self._embedding_coords is not None else (None, None)

    def setup_plot(self):
        self.graph.reset_graph()

    # Selection
    def apply_selection(self):
        if self.data is not None and self.__pending_selection is not None:
            selection = np.array(self.__pending_selection, dtype=np.uint8)
            self.graph.selection = selection
            self.__pending_selection = None
            self.graph.update_selection_colors()

    def selection_changed(self):
        graph_sel = self.graph.selection
        self.selection = list(graph_sel) if graph_sel is not None else None
        self.commit()

    # Output
    def commit(self):
        self.send_data()

    def send_data(self):
        group_sel, data, graph = None, self._get_projection_data(), self.graph
        if graph.selection is not None:
            group_sel = np.zeros(len(data), dtype=int)
            group_sel[self.valid_data] = graph.selection
        self.Outputs.selected_data.send(
            self._get_selected_data(data, graph.get_selection(), group_sel))
        self.Outputs.annotated_data.send(
            self._get_annotated_data(data, graph.get_selection(), group_sel,
                                     graph.selection))

    def _get_projection_data(self):
        if self.embedding_all_data is None:
            return self.data
        variables = self._get_projection_variables()
        data = self.data.transform(Domain(self.data.domain.attributes,
                                          self.data.domain.class_vars,
                                          self.data.domain.metas + variables))
        data.metas[:, -2:] = self.embedding_all_data
        return data

    def _get_projection_variables(self):
        domain = self.data.domain
        names = get_unique_names(
            [v.name for v in domain.variables + domain.metas],
            self.embedding_variables_names
        )
        return ContinuousVariable(names[0]), ContinuousVariable(names[1])

    @staticmethod
    def _get_selected_data(data, selection, group_sel):
        return create_groups_table(data, group_sel, False, "Group") \
            if len(selection) else None

    @staticmethod
    def _get_annotated_data(data, selection, group_sel, graph_sel):
        if graph_sel is not None and np.max(graph_sel) > 1:
            return create_groups_table(data, group_sel)
        else:
            return create_annotated_table(data, selection)

    # Report
    def send_report(self):
        if self.data is None:
            return

        caption = self._get_send_report_caption()
        self.report_plot()
        if caption:
            self.report_caption(caption)

    def _get_send_report_caption(self):
        return report.render_items_vert((
            ("Color", self._get_caption_var_name(self.attr_color)),
            ("Label", self._get_caption_var_name(self.attr_label)),
            ("Shape", self._get_caption_var_name(self.attr_shape)),
            ("Size", self._get_caption_var_name(self.attr_size)),
            ("Jittering", self.graph.jitter_size != 0 and
             "{} %".format(self.graph.jitter_size))))

    @staticmethod
    def _get_caption_var_name(var):
        return var.name if isinstance(var, Variable) else var

    # Misc
    def sizeHint(self):
        return QSize(1132, 708)

    def clear(self):
        self.data = None
        self.valid_data = None
        self.selection = None
        self._embedding_coords = None
        self.graph.clear()

    def onDeleteWidget(self):
        super().onDeleteWidget()
        self.graph.plot_widget.getViewBox().deleteLater()
        self.graph.plot_widget.clear()


if __name__ == "__main__":
    class OWProjectionWidgetWithName(OWProjectionWidget):
        name = "projection"

    app = QApplication([])
    ow = OWProjectionWidgetWithName()
    table = Table("iris")
    ow.set_data(table)
    ow.set_subset_data(table[::10])
    ow.handleNewSignals()
    ow.show()
    app.exec_()
    ow.saveSettings()
