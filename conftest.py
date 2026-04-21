import os

# Suppress duplicate OpenMP runtime crash on macOS (and Windows) when PyTorch's
# bundled libiomp5 is loaded alongside libomp from conda-forge packages such as
# numpy or scikit-image.  Setting this before any torch import allows both
# runtimes to coexist.  It is a workaround rather than a true fix; the
# underlying cause is that no single conda-provided OpenMP library is shared
# across all packages in the environment.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
