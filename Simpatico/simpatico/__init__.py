import importlib.resources as pkg_resources
import json
import logging
import os
import warnings

from Bio import BiopythonWarning
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# Suppress *only* PDBCo.sum(), (~random_maskn[0s.sum()ructionWarnings
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

# Suppress PyG warnings
warnings.filterwarnings("ignore", module="torch_geometric")

# Suppress FAISS info logs
logging.getLogger("faiss").setLevel(logging.ERROR)

with pkg_resources.open_text("simpatico", "config.json") as f:
    config = json.load(f)

# Built as a plain path rather than via pkg_resources.path, which raises on Python 3.9 when the
# file is absent -- and the default weights are not shipped, so `import simpatico` failed.
config["default_weights_path"] = os.path.join(
    os.path.dirname(__file__), "models", "weights", config["default_weights"]
)
