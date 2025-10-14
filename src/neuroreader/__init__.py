__all__ = ("NEV", "NFx", "NSx")

import pint
import pint_xarray

from neuroreader._nev import NEV
from neuroreader._nsx_nfx import NFx, NSx

ureg = pint.UnitRegistry()
ureg = pint_xarray.setup_registry(ureg)
ureg.formatter.default_format = "~P"

pint.set_application_registry(ureg)
