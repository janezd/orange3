from functools import reduce
from typing import Callable

from AnyQt.QtCore import Qt
from AnyQt.QtGui import QFontMetrics
from AnyQt.QtWidgets import QWidget, QSlider

from Orange.data import ContinuousVariable, DiscreteVariable
from Orange.preprocess import short_time_units, time_binnings, decimal_binnings
from Orange.preprocess.discretize import Discretizer
from orangewidget import gui


class VariableBinner:
    """
    Class for controlling a slider for binning variables

    To use it, the widget has to create a setting called `binner_settings`,
    with default value `{}`. Then insert a slider like:

    ```
    self.binner = VariableBinner(widget, self)
    ```

    If you use multiple binners in the widget, also provide a `binner_id`
    argument (str or int).

    Additional arguments, like callbacks and labels and also be provided.

    When the variables to which the binner refers is changed, extract the
    corresponding column (e.g. with obj:`Table.get_column_view`) and call

    ```
    self.binner.recompute_binnings(column, is_time)
    ```

    where `is_time` indicates whether this is a time variable.

    To obtain the binned variable, call

    ```
    attr = self.binner.binned_var(self.original_variable)
    ```

    typically followed by

    ```
    binned_data = attr.compute_value(data)
    ```

    to obtain binned data.
    """
    def __init__(self, widget: QWidget, master: OWWidget,
                 callback: Callable[[OWWidget], None],
                 on_released: Callable[[OWWidget], None],
                 hide_when_inactive: bool = False,
                 label: str = "Bin width",
                 binner_id: Union[str, int] = None):
        self.master = master
        self.binnings: List[BinDefinition] = []
        self.hide_when_inactive = hide_when_inactive
        self.binner_id = binner_id

        self.box = self.slider = self.bin_width_label = None
        self.setup_gui(widget, label)
        assert self.box and self.slider and self.bin_width_label

        self.slider.sliderMoved.connect(self._set_bin_width_slider_label)
        if self.callback:
            self.slider.sliderMoved.connect(callback)
        self.slider.sliderReleased.connect(on_released)
        self.master.settingsAboutToBePacked.connect(self._pack_settings)

    def setup_gui(self, widget: QWidget, label: str):
        """
        Create slider and label. Override for a different layout

        Args:
            widget (QWidget): the place where to insert the components
            label: label to the left of the slider
        """
        self.box = gui.hBox(widget)
        gui.widgetLabel(self.box, label)
        self.slider = QSlider(Qt.Horizontal)
        self.box.layout().addWidget(self.slider)
        self.bin_width_label = gui.widgetLabel(self.box)
        self.bin_width_label.setFixedWidth(35)
        self.bin_width_label.setAlignment(Qt.AlignRight)

    def _pack_settings(self):
        if self.binnings:
            self.master.binner_settings[self.binner_id] = self.bin_index

    @staticmethod
    def _short_text(label):
        return reduce(
            lambda s, rep: s.replace(*rep),
            short_time_units.items(), label)

    def _set_bin_width_slider_label(self):
        if self.bin_index < len(self.binnings):
            text = self._short_text(
                self.binnings[self.bin_index].width_label)
        else:
            text = ""
        self.bin_width_label.setText(text)

    @property
    def bin_index(self) -> int:
        """Index of currently selected entry in binnings; for internal use"""
        return self.slider.value()

    @bin_index.setter
    def bin_index(self, value: int):
        self.slider.setValue(value)

    def recompute_binnings(self, column: np.ndarray, is_time: bool,
                           **binning_args):
        """
        Recomputes the set of available binnings based on data

        The method accepts the same keyword arguments as
        :obj:`Orange.preprocess.discretize.decimal_binnings` and
        :obj:`Orange.preprocess.discretize.time_binnings`.

        Args:
            column (np.ndarray): column with data for binning
            is_time (bool): indicates whether this is a time variable
            **binning_args: see documentation for
                :obj:`Orange.preprocess.discretize.decimal_binnings` and
                :obj:`Orange.preprocess.discretize.time_binnings`.
        """
        inactive = column is None or not np.any(np.isfinite(column))
        if self.hide_when_inactive:
            self.box.setVisible(not inactive)
        else:
            self.box.setDisabled(inactive)
        if inactive:
            self.binnings = []
            return

        if is_time:
            self.binnings = time_binnings(column, **binning_args)
        else:
            self.binnings = decimal_binnings(column, **binning_args)
        fm = QFontMetrics(self.master.font())
        width = max(fm.size(Qt.TextSingleLine,
                            self._short_text(binning.width_label)
                            ).width()
                    for binning in self.binnings)
        self.bin_width_label.setFixedWidth(width)
        max_bins = len(self.binnings) - 1
        self.slider.setMaximum(max_bins)

        pending = self.master.binner_settings.get(self.binner_id, None)
        if pending is not None:
            self.bin_index = pending
            del self.master.binner_settings[self.binner_id]
        if self.bin_index > max_bins:
            self.bin_index = max_bins
        self._set_bin_width_slider_label()

    def current_binning(self) -> BinDefinition:
        """Return the currently selected binning"""
        return self.binnings[self.bin_index]

    def binned_var(self, var: ContinuousVariable) -> DiscreteVariable:
        """
        Creates a discrete variable for the given continuous variable,
        using the currently selected binning
        """
        binning = self.binnings[self.bin_index]
        discretizer = Discretizer(var, list(binning.thresholds[1:-1]))
        blabels = binning.labels[1:-1]
        labels = [f"< {blabels[0]}"] + [
            f"{lab1} - {lab2}" for lab1, lab2 in zip(blabels, blabels[1:])
        ] + [f"≥ {blabels[-1]}"]
        return DiscreteVariable(
            name=var.name, values=labels, compute_value=discretizer)
