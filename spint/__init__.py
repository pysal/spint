__version__ = '1.0.7'

from .gravity import Gravity, Production, Attraction, Doubly
from .utils import CPC, sorensen, srmse
from .vec_SA import VecMoran as Moran_Vector
from .dispersion import phi_disp, alpha_disp
from .generate_dummy_accessibility import generate_dummy_flows
from .flow_accessibility import Accessibility, AFED
