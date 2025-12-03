import functools
import os
from pathlib import Path
from typing import ClassVar, Literal, final

import numpy as np
import numpy.typing as npt
import pandas as pd
import pint_pandas  # noqa: F401

from neuroreader._utilities import (
    DataPacket,
    Field,
    Header,
    parse_file_spec,
    parse_filter_details,
    parse_time_origin,
    parse_time_resolution,
    read_fields,
)

CLOCK_FREQUENCY_IN_HZ = 30_000

BASIC_HEADER_FIELDS = (
    Field("File Type ID", "<", "s", 8),
    Field("File Spec", "<", "B", 2),
    Field("Bytes in Headers", "<", "I", 4),
    Field("Label", "<", "s", 16),
    Field("Comments", "<", "s", 200),
    Field("Application to Create File", "<", "s", 52),
    Field("Processor Timestamp", "<", "I", 4),
    Field("Period", "<", "I", 4),
    Field("Time Resolution of Time Stamps", "<", "I", 4),
    Field("Time Origin", "<", "H", 16),
    Field("Channel Count", "<", "I", 4),
)

EXTENDED_HEADER_FIELDS = (
    Field("Type", "<", "s", 2),
    Field("Electrode ID", "<", "H", 2),
    Field("Electrode label", "<", "s", 16),
    Field("Front End ID", "<", "B", 1),
    Field("Front End Connector Pin", "<", "B", 1),
    Field("Min Digital Value", "<", "h", 2),
    Field("Max Digital Value", "<", "h", 2),
    Field("Min Analog Value", "<", "h", 2),
    Field("Max Analog Value", "<", "h", 2),
    Field("Units", "<", "s", 16),
    Field("High Pass Corner Frequency", "<", "I", 4),
    Field("High Pass Filter Order", "<", "I", 4),
    Field("High Pass Filter Type", "<", "H", 2),
    Field("Low Pass Corner Frequency", "<", "I", 4),
    Field("Low Pass Filter Order", "<", "I", 4),
    Field("Low Pass Filter Type", "<", "H", 2),
)

DATA_PACKET_FIELDS = (
    Field("Header", "<", "B", 1),
    Field("Timestamp", "<", "I", 4),
    Field("Number of Data Points", "<", "I", 4),
)


class _NFxOrNSx:
    _FILETYPE: ClassVar[Literal["NSx", "NFx"]]

    def __init__(self, filepath: Path, *, dtype: npt.DTypeLike = np.float32) -> None:
        self._filepath = filepath
        self.dtype = dtype
        self._read_headers()

    @functools.cached_property
    def basic_header(self) -> Header:
        return self._basic_header

    @functools.cached_property
    def extended_headers(self) -> pd.DataFrame:
        return self._extended_headers

    @functools.cached_property
    def data(self) -> list[DataPacket]:
        self._read_data_packets()
        return self._data

    def _read_headers(self) -> None:
        with self._filepath.open("rb") as f:
            self._basic_header = _parse_basic_header(
                read_fields(f, fields=BASIC_HEADER_FIELDS),
            )

            self._extended_headers = _parse_extended_headers(
                [
                    read_fields(f, fields=EXTENDED_HEADER_FIELDS)
                    for _ in range(self._basic_header["Channel Count"])
                ],
            )

    def _read_data_packets(self) -> None:
        n_channels: int = self._basic_header["Channel Count"]

        match self._FILETYPE:
            case "NSx":
                dtype = np.dtype(np.int16)
            case "NFx":
                dtype = np.dtype(np.float32)

        data_packets: list[DataPacket] = []

        with self._filepath.open("rb") as f:
            file_size = f.seek(0, os.SEEK_END)
            _ = f.seek(self._basic_header["Bytes in Headers"], os.SEEK_SET)

            while f.tell() < file_size:
                packet: DataPacket = read_fields(f, fields=DATA_PACKET_FIELDS)

                if packet["Header"] != 1:
                    break

                n_data_points = packet["Number of Data Points"]

                packet["Data Points"] = np.memmap(
                    self._filepath,
                    dtype=dtype.newbyteorder("<"),
                    mode="r",
                    offset=f.tell(),
                    shape=(n_data_points, n_channels),
                )
                _ = f.seek(n_data_points * n_channels * dtype.itemsize, os.SEEK_CUR)
                data_packets.append(packet)

        self._data = data_packets

    def sampling_frequency(self) -> float:
        return CLOCK_FREQUENCY_IN_HZ / self.basic_header["Period"]


def _parse_basic_header(x: Header, /) -> Header:
    return x | {
        "File Spec": parse_file_spec(x["File Spec"]),
        "Time Origin": parse_time_origin(x["Time Origin"]),
        "Time Resolution of Time Stamps": parse_time_resolution(
            x["Time Resolution of Time Stamps"],
        ),
    }


def _parse_extended_headers(
    extended_headers: list[Header],
) -> pd.DataFrame:
    headers = pd.DataFrame(extended_headers)
    headers = parse_filter_details(headers)
    return (
        headers.assign(**{
            "Neural Processor Port": lambda x: (
                x["Front End ID"]
                .replace(list(range(4)), "A")
                .replace(list(range(4, 8)), "B")
                .replace(list(range(8, 12)), "C")
                .replace(list(range(12, 16)), "D")
            ),
            "Analog Data Channel": lambda x: (x["Electrode ID"] >= 10_241),
            "Recording Electrode": lambda x: ~x["Analog Data Channel"],
        }).astype({
            "Type": "string",
            "Electrode ID": np.uint16,
            "Electrode label": "string",
            "Front End ID": np.uint8,
            "Front End Connector Pin": np.uint8,
            "Min Digital Value": np.int16,
            "Max Digital Value": np.int16,
            "Min Analog Value": np.int16,
            "Max Analog Value": np.int16,
            "Units": "string",
            "Neural Processor Port": pd.CategoricalDtype(["A", "B", "C", "D"]),
        })
    ).set_index("Electrode ID")


@final
class NFx(_NFxOrNSx):
    _FILETYPE = "NFx"

    @property
    def NEUCDFLT(self) -> Header:  # noqa: N802
        return self.basic_header

    @property
    def FC(self) -> Header:  # noqa: N802
        return self.extended_headers


@final
class NSx(_NFxOrNSx):
    _FILETYPE = "NSx"

    @property
    def NEURALCD(self) -> Header:  # noqa: N802
        return self.basic_header

    @property
    def CC(self) -> Header:  # noqa: N802
        return self.extended_headers
