import sys
from pathlib import Path
from unittest.mock import MagicMock

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# loading_tiff_files_prototyping imports caiman/tifffile at module level.
# Stub them out if they aren't installed so tests can still exercise the
# pure filesystem logic (generate_file_list, create_output_directory)
# without requiring the full caiman conda environment.
for module_name in (
    "caiman",
    "caiman.source_extraction",
    "caiman.source_extraction.cnmf",
    "caiman.paths",
    "tifffile",
):
    try:
        __import__(module_name)
    except ImportError:
        sys.modules[module_name] = MagicMock()
