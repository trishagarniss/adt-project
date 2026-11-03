import os
import sys
from pathlib import Path

# Setup temp paths to avoid permission issues
temp_dir = Path(os.environ.get("TEMP", "C:/temp"))
os.environ["JUPYTER_CONFIG_DIR"] = str(temp_dir / "jupyter_config")
os.environ["JUPYTER_DATA_DIR"] = str(temp_dir / "jupyter_data")
os.environ["JUPYTER_RUNTIME_DIR"] = str(temp_dir / "jupyter_runtime")
os.environ["MPLCONFIGDIR"] = str(temp_dir / "mpl")
os.environ["IPYTHONDIR"] = str(temp_dir / "ipython")

for p in [os.environ["JUPYTER_CONFIG_DIR"], os.environ["JUPYTER_DATA_DIR"], 
          os.environ["JUPYTER_RUNTIME_DIR"], os.environ["MPLCONFIGDIR"], os.environ["IPYTHONDIR"]]:
    Path(p).mkdir(parents=True, exist_ok=True)

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

nb_path = Path("notebooks") / "01_eksplorasi_dan_prapemrosesan_kunto.ipynb"
print(f"Reading {nb_path}...")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
print("Executing notebook cells...")
ep.preprocess(nb, {"metadata": {"path": str(Path("notebooks").resolve())}})

with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook executed successfully and saved to {nb_path}!")

