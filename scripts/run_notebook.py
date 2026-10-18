"""Execute a notebook and save its outputs in the notebook itself."""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('notebook', nargs='?', default='notebooks/baseline/01_hotel_booking_patterns.ipynb')
args = parser.parse_args()
path = ROOT / args.notebook
for key, subdir in [('MPLCONFIGDIR', '.mpl'), ('IPYTHONDIR', '.ipython'),
                    ('JUPYTER_RUNTIME_DIR', '.jupyter-runtime')]:
    directory = ROOT / '.cache/notebook_runner' / subdir
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(directory)
os.environ['MPLBACKEND'] = 'module://matplotlib_inline.backend_inline'

import nbformat
from nbclient import NotebookClient

notebook = nbformat.read(path, as_version=4)
client = NotebookClient(notebook, timeout=180, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}})
client.execute()
nbformat.validate(notebook)
# NotebookClient stops on the first cell error, before replacing saved outputs.
nbformat.write(notebook, path)
print(f'Executed {sum(c.cell_type == "code" for c in notebook.cells)} code cells without errors.')
print(f'Saved notebook: {path}')
