import os
import sys
from pathlib import Path

# Setup temp paths to avoid permission issues on Windows
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
from nbclient import NotebookClient

nb_path = Path("notebooks") / "03_moving_average_dan_exponential_smoothing_maulana.ipynb"
print(f"Reading {nb_path}...")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(Path(".").resolve())}})
print("Executing notebook cells with NotebookClient...")
client.execute()

with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook executed successfully and saved to {nb_path}!")
