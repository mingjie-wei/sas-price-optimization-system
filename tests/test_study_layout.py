"""A recorded path relocation must not silently authorize later source edits."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import study_layout

class StudyLayoutTests(unittest.TestCase):
    def test_only_exact_recorded_source_keeps_historical_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'runner.py';source.write_text('relocated path')
            actual=hashlib.sha256(source.read_bytes()).hexdigest()
            receipt=root/'docs/provenance/study_layout.json';receipt.parent.mkdir(parents=True)
            receipt.write_text(json.dumps({'source_relocations':{'runner.py':{
                'original_sha256':'historical-identity','relocated_sha256':actual,'historical_path':'old/runner.py'}}}))
            with patch.object(study_layout,'ROOT',root):
                self.assertEqual(study_layout.training_source_digest(source),'historical-identity')
                self.assertEqual(study_layout.historical_source_path(source),'old/runner.py')
                source.write_text('a later training change')
                changed=hashlib.sha256(source.read_bytes()).hexdigest()
                self.assertEqual(study_layout.training_source_digest(source),changed)
                self.assertNotEqual(changed,'historical-identity')
                self.assertEqual(study_layout.historical_source_path(source),'old/runner.py')

    def test_unregistered_source_uses_actual_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'runner.py';source.write_text('new experiment')
            with patch.object(study_layout,'ROOT',root):
                self.assertEqual(study_layout.training_source_digest(source),hashlib.sha256(source.read_bytes()).hexdigest())
