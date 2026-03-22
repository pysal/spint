# ruff: noqa: F401 - imported but unused

import contextlib
from importlib.metadata import PackageNotFoundError, version

from .dispersion import alpha_disp, phi_disp
from .flow_accessibility import Accessibility
from .gravity import Attraction, Doubly, Gravity, Production
from .utils import (
    # CPC,  # problem -- `Y` not defined inside function -- inoperable
    sorensen,
    srmse,
)
from .vec_SA import VecMoran as Moran_Vector

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("spint")
