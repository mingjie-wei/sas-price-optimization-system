"""Run the editable notebook and save an HTML reading copy."""
import argparse
import os
import re
from urllib.parse import urlsplit, unquote, quote
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('notebook', nargs='?', default='notebooks/baseline/01_hotel_booking_patterns.ipynb')
parser.add_argument('--output')
parser.add_argument('--export-only', action='store_true', help='Keep saved outputs; refresh HTML only')
args = parser.parse_args()
path = ROOT / args.notebook
legacy_outputs = {
    '01_hotel_booking_patterns.ipynb': 'outputs/analysis/hotel_booking_patterns.html',
    '02_simulator_and_offline_data.ipynb': 'outputs/simulator/simulator_and_offline_data.html',
    '03_offline_cql.ipynb': 'outputs/baseline/cql/offline_cql.html',
    '04_end_to_end_demo.ipynb': 'outputs/story/end_to_end_demo.html',
    '04_end_to_end_story.ipynb': 'outputs/story/end_to_end_story.html',
}
def reading_copy(notebook_path):
    if notebook_path.parent.name == 'baseline':
        return ROOT / legacy_outputs[notebook_path.name]
    relative = notebook_path.relative_to(ROOT / 'notebooks')
    return (ROOT / 'outputs' / relative).with_suffix('.html')

html_path = ROOT / args.output if args.output else reading_copy(path)
OUT = html_path.parent
OUT.mkdir(parents=True, exist_ok=True)
for key, subdir in [('MPLCONFIGDIR', '.mpl'), ('IPYTHONDIR', '.ipython'),
                    ('JUPYTER_RUNTIME_DIR', '.jupyter-runtime')]:
    directory = OUT / subdir
    directory.mkdir(exist_ok=True)
    os.environ[key] = str(directory)
os.environ['MPLBACKEND'] = 'module://matplotlib_inline.backend_inline'

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

notebook = nbformat.read(path, as_version=4)
client = NotebookClient(notebook, timeout=180, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}})
if not args.export_only:
    client.execute()
nbformat.validate(notebook)
# NotebookClient stops on the first cell error, before replacing saved outputs.
nbformat.write(notebook, path)
exporter = HTMLExporter(template_name='lab')
exporter.exclude_input_prompt = True
exporter.exclude_output_prompt = True
exporter.exclude_input = not notebook.metadata.get('show_code', False)
body, _ = exporter.from_notebook_node(notebook)
# Resolve notebook-relative links before moving the page into outputs/.
def rewrite_link(match):
    href = match.group(1)
    parts = urlsplit(href)
    if parts.scheme or parts.netloc or not parts.path:
        return match.group(0)
    target = (path.parent / unquote(parts.path)).resolve()
    if target.suffix == '.ipynb' and target.exists():
        target = reading_copy(target)
    relative = quote(os.path.relpath(target, OUT), safe='/')
    if parts.fragment:
        relative += '#' + parts.fragment
    return f'href="{relative}"'

body = re.sub(r'href="([^"]+)"', rewrite_link, body)
html_path.write_text(body)
print('Exported saved outputs.' if args.export_only else f'Executed {sum(c.cell_type == "code" for c in notebook.cells)} code cells without errors.')
print(f'Reading copy: {html_path}')
