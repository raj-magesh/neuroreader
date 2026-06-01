import struct
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pint
import pint_pandas  # noqa: F401

if TYPE_CHECKING:
    from collections.abc import Sequence
    from io import BufferedReader

Header = dict[str, Any]
DataPacket = dict[str, Any]

ureg = pint.UnitRegistry()
ureg.formatter.default_format = "~P"

pint.set_application_registry(ureg)

_STRUCT_FORMAT_TO_NUMPY_DTYPE_MAPPING = {
    "I": "u4",
    "i": "i4",
    "H": "u2",
    "h": "i2",
    "B": "B",
}


class Field(NamedTuple):
    name: str
    prefix: Literal["@", "=", "<", ">", "!"]
    format_character: Literal["c", "b", "B", "h", "H", "i", "I", "f", "d", "s"]
    n_bytes: int

    def to_numpy_dtype(self) -> npt.DTypeLike:
        label = (self.name, self.name.lower().replace(" ", "_"))
        format_string = f"{self.prefix}{_STRUCT_FORMAT_TO_NUMPY_DTYPE_MAPPING[self.format_character]}"
        n_elements = self.n_bytes // struct.calcsize(self.format_character)

        if n_elements == 1:
            return (label, format_string)
        return (label, format_string, n_elements)


def read_field(
    f: BufferedReader,
    *,
    field: Field,
    extract_single_element_tuple: bool = True,
    decode_strings: bool = True,
    strip_null_bytes: bool = True,
) -> str | int | float | bytes:
    output = struct.unpack(
        f"{field.prefix}{field.n_bytes // struct.calcsize(field.format_character)}{field.format_character}",
        f.read(field.n_bytes),
    )

    if extract_single_element_tuple and len(output) == 1:
        output = output[0]

    if field.format_character == "s":
        if decode_strings:
            output = cast("bytes", output)
            output = output.decode("utf-8")
        if strip_null_bytes:
            output = cast("str", output)
            output = output.rstrip("\x00")

    return cast("str | int | float | bytes", output)


def read_fields(
    f: BufferedReader,
    *,
    fields: Sequence[Field],
    **kwargs,
) -> dict[str, str | int | float | bytes]:
    return {
        field.name: read_field(
            f,
            field=field,
            **kwargs,
        )
        for field in fields
    }


def parse_file_spec(spec: tuple[int, int]) -> str:
    return f"{spec[0]}.{spec[1]}"


def parse_time_origin(
    timestamp: tuple[int, int, int, int, int, int, int, int],
) -> datetime:
    return datetime(
        timestamp[0],
        timestamp[1],
        timestamp[3],
        timestamp[4],
        timestamp[5],
        timestamp[6],
        timestamp[7] * 1_000,
        tzinfo=UTC,
    )


def parse_time_resolution(resolution: int) -> pint.Quantity:
    return pint.Quantity(resolution, "Hz")


def parse_filter_details(header: pd.DataFrame) -> pd.DataFrame:
    filter_dtype = pd.CategoricalDtype(
        categories=["None", "Butterworth", "Chebyshev"],
    )

    return header.assign(**{
        f"{direction} Pass Filter Type": (
            pd.col(f"{direction} Pass Filter Type").replace({
                0: "None",
                1: "Butterworth",
                2: "Chebyshev",
            })
        )
        for direction in ("High", "Low")
    }).astype({
        "High Pass Corner Frequency": "pint[mHz][UInt32]",
        "High Pass Filter Order": np.uint32,
        "High Pass Filter Type": filter_dtype,
        "Low Pass Corner Frequency": "pint[mHz][UInt32]",
        "Low Pass Filter Order": np.uint32,
        "Low Pass Filter Type": filter_dtype,
    })
