__all__ = ("CLOCK_FREQUENCY_IN_HZ", "NEV", "NFx", "NSx")

import pint

from neuroreader._nev import NEV
from neuroreader._nsx_nfx import CLOCK_FREQUENCY_IN_HZ, NFx, NSx

ureg = pint.UnitRegistry()
ureg.formatter.default_format = "~P"

pint.set_application_registry(ureg)
