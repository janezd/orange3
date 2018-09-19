# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import numpy as np
import scipy.sparse as sp

from AnyQt.QtCore import QRectF, QPointF

from Orange.data import Table
from Orange.widgets.tests.base import (
    WidgetTest, WidgetOutputsTestMixin, ProjectionWidgetTestMixin
)
from Orange.widgets.visualize.utils.widget import OWProjectionWidget


class TestOWProjectionWidget(WidgetTest, WidgetOutputsTestMixin,
                             ProjectionWidgetTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = "Data"
        cls.signal_data = cls.data
        cls.same_input_output_domain = False

    def setUp(self):
        self.widget = self.create_widget(OWProjectionWidget)

    def _select_data(self):
        self.widget.graph.select_by_rectangle(
            QRectF(QPointF(-20, -20), QPointF(20, 20)))
        return self.widget.graph.get_selection()

    def _compare_selected_annotated_domains(self, selected, annotated):
        selected_vars = selected.domain.variables
        annotated_vars = annotated.domain.variables
        self.assertLessEqual(set(selected_vars), set(annotated_vars))

    def test_saved_selection(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.graph.select_by_index(list(range(0, len(self.data), 10)))
        settings = self.widget.settingsHandler.pack_data(self.widget)
        w = self.create_widget(OWProjectionWidget, stored_settings=settings)
        self.send_signal(self.widget.Inputs.data, self.data, widget=w)
        np.testing.assert_equal(self.widget.graph.selection, w.graph.selection)

    def test_sparse(self):
        table = Table("iris")
        table.X = sp.csr_matrix(table.X)
        self.assertTrue(sp.issparse(table.X))
        self.send_signal(self.widget.Inputs.data, table)
        self.assertTrue(self.widget.Error.sparse_data.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.sparse_data.is_shown())
