import contextlib
from importlib.metadata import PackageNotFoundError, version

from .gravity import Gravity, Production, Attraction, Doubly
from .utils import CPC, sorensen, srmse
from .vec_SA import VecMoran as Moran_Vector
from .dispersion import phi_disp, alpha_disp
from .flow_accessibility import Accessibility

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("spint")
