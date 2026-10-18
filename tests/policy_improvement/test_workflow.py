"""Public commands preserve the order and boundaries of the research workflow."""
import unittest
from unittest.mock import patch
from scripts.policy_improvement import experiments, confirmation, report


class WorkflowTests(unittest.TestCase):
    def test_report_verifies_evidence_before_writing(self):
        calls = []
        with patch('sys.argv', ['report']), \
             patch.object(report, 'summarize_experiments', side_effect=lambda: calls.append('diagnostics')), \
             patch.object(report, 'build_evidence', side_effect=lambda: calls.append('evidence')), \
             patch.object(report, 'build_report', side_effect=lambda: calls.append('report')):
            report.main()
        self.assertEqual(calls, ['evidence', 'report'])

    def test_full_report_refreshes_diagnostics_first(self):
        calls = []
        with patch('sys.argv', ['report', '--refresh-diagnostics']), \
             patch.object(report, 'summarize_experiments', side_effect=lambda: calls.append('diagnostics')), \
             patch.object(report, 'build_evidence', side_effect=lambda: calls.append('evidence')), \
             patch.object(report, 'build_report', side_effect=lambda: calls.append('report')):
            report.main()
        self.assertEqual(calls, ['diagnostics', 'evidence', 'report'])

    def test_confirmation_uses_fresh_data_workflow(self):
        with patch('sys.argv', ['experiments', 'confirmation']), \
             patch.object(experiments, 'initialize', return_value={}), \
             patch.object(experiments, 'protect') as protect, \
             patch.object(confirmation, 'main') as confirm:
            experiments.main()
        confirm.assert_called_once_with()
        protect.assert_called_once_with({})
