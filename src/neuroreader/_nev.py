import functools
import itertools
import os
from typing import TYPE_CHECKING, Literal, TypedDict, final

import numpy as np
import numpy.typing as npt
import pandas as pd
import pint_pandas  # noqa: F401
import xarray as xr

from neuroreader._utilities import (
    Field,
    Header,
    parse_file_spec,
    parse_filter_details,
    parse_time_origin,
    parse_time_resolution,
    read_field,
    read_fields,
)

if TYPE_CHECKING:
    from pathlib import Path

Events = TypedDict(
    "Events",
    {
        "Digital Events": pd.DataFrame | None,
        "Spike Events": xr.DataArray | None,
        "Stimulation Events": xr.DataArray | None,
    },
)

BASIC_HEADER_FIELDS = (
    Field("File Type ID", "<", "s", 8),
    Field("File Spec", "<", "B", 2),
    Field("Additional Flags", "<", "H", 2),
    Field("Bytes in Headers", "<", "I", 4),
    Field("Bytes in Data Packets", "<", "I", 4),
    Field("Time Resolution of Time Stamps", "<", "I", 4),
    Field("Time Resolution of Samples", "<", "I", 4),
    Field("Time Origin", "<", "H", 16),
    Field("Application to Create File", "<", "s", 32),
    Field("Comment Field", "<", "s", 200),
    Field("Reserved", "<", "s", 52),
    Field("Processor Timestamp", "<", "I", 4),
    Field("Number of Extended Headers", "<", "I", 4),
)

EXTENDED_HEADER_FIELDS: dict[str, tuple[Field, ...]] = {
    "NEUEVWAV": (
        Field("Packet ID", "<", "s", 8),
        Field("Electrode ID", "<", "H", 2),
        Field("Front End ID", "<", "B", 1),
        Field("Front End Connector Pin", "<", "B", 1),
        Field("Neural Amp Digitization Factor", "<", "H", 2),
        Field("Energy Threshold", "<", "H", 2),
        Field("High Threshold", "<", "h", 2),
        Field("Low Threshold", "<", "h", 2),
        Field("Number of Sorted Units", "<", "B", 1),
        Field("Bytes per Sample", "<", "B", 1),
        Field("Stim Amp Digitization Factor", "<", "f", 4),
        Field("Reserved", "<", "s", 6),
    ),
    "NEUEVFLT": (
        Field("Packet ID", "<", "s", 8),
        Field("Electrode ID", "<", "H", 2),
        Field("High Pass Corner Frequency", "<", "I", 4),
        Field("High Pass Filter Order", "<", "I", 4),
        Field("High Pass Filter Type", "<", "H", 2),
        Field("Low Pass Corner Frequency", "<", "I", 4),
        Field("Low Pass Filter Order", "<", "I", 4),
        Field("Low Pass Filter Type", "<", "H", 2),
        Field("Reserved", "<", "s", 2),
    ),
    "NEUEVLBL": (
        Field("Packet ID", "<", "s", 8),
        Field("Electrode ID", "<", "H", 2),
        Field("Label", "<", "s", 16),
        Field("Reserved", "<", "s", 6),
    ),
    "DIGLABEL": (
        Field("Packet ID", "<", "s", 8),
        Field("Label", "<", "s", 16),
        Field("Mode", "<", "B", 1),
        Field("Reserved", "<", "s", 7),
    ),
}

DATA_PACKET_ID_MAPPING = {
    "Digital Events": [0],
    "Spike Events": list(range(1, 512 + 1)),
    "Stimulation Events": list(range(5121, 5632 + 1)),
}

DATA_PACKET_FIELDS: dict[str, list[Field]] = {
    "Digital Events": [
        Field("Timestamp", "<", "I", 4),
        Field("Packet ID", "<", "H", 2),
        Field("Packet Insertion Reason", "<", "B", 1),
        Field("Reserved", "<", "B", 1),
        Field("Parallel Input", "<", "H", 2),
        Field("SMA Input 1", "<", "h", 2),
        Field("SMA Input 2", "<", "h", 2),
        Field("SMA Input 3", "<", "h", 2),
        Field("SMA Input 4", "<", "h", 2),
        # should have Field("Reserved", "<", "B", packet_size - 18)
        # but packet_size is variable, so computed dynamically
    ],
    "Spike Events": [
        Field("Timestamp", "<", "I", 4),
        Field("Packet ID", "<", "H", 2),
        Field("Unit Classification Number", "<", "B", 1),
        Field("Reserved", "<", "B", 1),
        # should have Field("Reserved", "<", "B", packet_size - 8)
        # but packet_size is variable, so computed dynamically
    ],
    "Stimulation Events": [
        Field("Timestamp", "<", "I", 4),
        Field("Packet ID", "<", "H", 2),
        Field("Reserved", "<", "B", 2),
        # should have Field("Reserved", "<", "B", packet_size - 8)
        # but packet_size is variable, so computed dynamically
    ],
}


@final
class NEV:
    def __init__(self, filepath: Path, *, n_packets_per_buffer: int = 2**20) -> None:
        self._filepath = filepath
        self._data: Events
        self._read_headers()
        self._n_packets_per_buffer = n_packets_per_buffer

    @functools.cached_property
    def basic_header(self) -> Header:
        return self._basic_header

    @functools.cached_property
    def extended_headers(self) -> dict[str, pd.DataFrame]:
        return self._extended_headers

    @functools.cached_property
    def data(self) -> Events:
        self._read_data_packets()
        return self._data

    @property
    def NEURALEV(self) -> Header:  # noqa: N802
        return self._basic_header

    @property
    def NEUEVWAV(self) -> pd.DataFrame:  # noqa: N802
        return self.extended_headers["NEUEVWAV"]

    @property
    def NEUEVFLT(self) -> pd.DataFrame:  # noqa: N802
        return self.extended_headers["NEUEVFLT"]

    @property
    def NEUEVLBL(self) -> pd.DataFrame:  # noqa: N802
        return self.extended_headers["NEUEVLBL"]

    @property
    def DIGLABEL(self) -> pd.DataFrame:  # noqa: N802
        return self.extended_headers["DIGLABEL"]

    @property
    def spikes(self) -> xr.DataArray | None:
        return self.data["Spike Events"]

    @property
    def stimulations(self) -> xr.DataArray | None:
        return self.data["Stimulation Events"]

    @property
    def digital_events(self) -> pd.DataFrame | None:
        return self.data["Digital Events"]

    def _read_headers(self) -> None:
        packet_id = Field("", "<", "s", 8)
        with self._filepath.open("rb") as f:
            self._basic_header = _parse_basic_header(
                read_fields(f, fields=BASIC_HEADER_FIELDS),
            )

            headers: list[Header] = []
            for _ in range(self._basic_header["Number of Extended Headers"]):
                packet_id_ = read_field(f, field=packet_id)
                _ = f.seek(-packet_id.n_bytes, os.SEEK_CUR)

                headers.append(
                    read_fields(f, fields=EXTENDED_HEADER_FIELDS[packet_id_]),
                )

            self._extended_headers = _parse_extended_headers(headers)

    def _read_data_packets(self) -> None:
        n_bytes_per_packet = self.basic_header["Bytes in Data Packets"]
        n_bytes_in_headers = self.basic_header["Bytes in Headers"]

        n_bytes_in_file = self._filepath.stat().st_size

        n_packets = (n_bytes_in_file - n_bytes_in_headers) / n_bytes_per_packet

        if n_packets.is_integer():
            n_packets = int(n_packets)
        else:
            raise ValueError

        self._data = {k: [] for k in DATA_PACKET_ID_MAPPING}

        for bytes_ in itertools.batched(
            range(n_bytes_in_headers, n_bytes_in_file),
            n=self._n_packets_per_buffer * n_bytes_per_packet,
            strict=False,
        ):
            contents = np.memmap(
                self._filepath,
                dtype=np.dtype("B"),
                mode="r",
                offset=bytes_[0],
                shape=(len(bytes_) // n_bytes_per_packet, n_bytes_per_packet),
            )

            packet_ids = np.squeeze(contents[:, 4:6].view("<u2"))
            if int("0xffffffff", 0) in packet_ids:
                error = "continuation packets not implemented"
                raise NotImplementedError(error)

            for event_type, ids in DATA_PACKET_ID_MAPPING.items():
                self._data[event_type].append(
                    contents[np.isin(packet_ids, ids), :],
                )

        for key, value in self._data.items():
            if len(value) != 0:
                self._data[key] = np.concatenate(value, axis=0)

        self._data["Digital Events"] = _parse_digital_events(
            self._data["Digital Events"],
            packet_size=n_bytes_per_packet,
        )
        for event_type in ("Spike Events", "Stimulation Events"):
            self._data[event_type] = _parse_spike_or_stimulation_events(
                self._data[event_type],
                event_type=event_type,
                header=self.NEUEVWAV,
                packet_size=n_bytes_per_packet,
            )


def _parse_basic_header(x: Header) -> Header:
    _ = x.pop("Reserved")
    return x | {
        "File Spec": parse_file_spec(x["File Spec"]),
        "Time Origin": parse_time_origin(x["Time Origin"]),
        "Time Resolution of Time Stamps": parse_time_resolution(
            x["Time Resolution of Time Stamps"],
        ),
        "Time Resolution of Samples": parse_time_resolution(
            x["Time Resolution of Samples"],
        ),
    }


def _parse_extended_headers(extended_headers: list[Header]) -> dict[str, pd.DataFrame]:
    headers = {key: [] for key in ("NEUEVWAV", "NEUEVFLT", "NEUEVLBL", "DIGLABEL")}

    for header in extended_headers:
        headers[header["Packet ID"]].append(header)

    headers = {key: pd.DataFrame(values) for key, values in headers.items()}

    headers["NEUEVWAV"] = (
        headers["NEUEVWAV"]
        .astype({
            "Electrode ID": np.uint16,
            "Front End ID": np.uint8,
            "Front End Connector Pin": np.uint8,
            "Neural Amp Digitization Factor": "pint[nV][UInt16]",
            "Energy Threshold": np.uint16,
            "High Threshold": "pint[uV][Int16]",
            "Low Threshold": "pint[uV][Int16]",
            "Number of Sorted Units": np.uint8,
            "Bytes per Sample": np.uint8,
            "Stim Amp Digitization Factor": "pint[V][Float32]",
            "Reserved": np.bytes_,
        })
        .assign(**{
            "Bytes per Sample": lambda x: x["Bytes per Sample"].replace({0: 1}),
        })
        .set_index("Electrode ID")
        .drop(columns=["Packet ID", "Reserved"])
    )

    headers["NEUEVFLT"] = parse_filter_details(headers["NEUEVFLT"])
    headers["NEUEVFLT"] = (
        headers["NEUEVFLT"]
        .astype({
            "Electrode ID": np.uint16,
            "Reserved": np.bytes_,
        })
        .set_index("Electrode ID")
        .drop(columns=["Packet ID", "Reserved"])
    )

    headers["NEUEVLBL"] = (
        headers["NEUEVLBL"]
        .astype({
            "Electrode ID": np.uint16,
            "Reserved": np.bytes_,
        })
        .set_index("Electrode ID")
        .drop(columns=["Packet ID", "Reserved"])
    )

    headers["DIGLABEL"] = (
        _parse_diglabel_mode(headers["DIGLABEL"])
        .drop(columns=["Packet ID", "Reserved"])
        .set_index("Label")
    )

    return headers


def _parse_spike_or_stimulation_events(
    events: npt.NDArray[np.uint8],
    *,
    event_type: Literal["Spike Events", "Stimulation Events"],
    header: Header,
    packet_size: int,
) -> xr.DataArray:
    parsed_events = []

    packet_ids = np.squeeze(events[:, 4:6].view("<u2"))
    for n_bytes_per_sample in pd.unique(header["Bytes per Sample"]):
        events_ = events[
            np.isin(
                packet_ids,
                header.loc[
                    header["Bytes per Sample"] == n_bytes_per_sample
                ].index.to_list(),
            ),
            ...,
        ]
        if len(events_) == 0:
            continue

        fields = DATA_PACKET_FIELDS[event_type]
        field_name = "Waveform"
        dtype = np.dtype(
            [field.to_numpy_dtype() for field in fields]
            + [
                (
                    (field_name, field_name.lower().replace(" ", "_")),
                    f"<i{n_bytes_per_sample}",
                    (packet_size - sum(field.n_bytes for field in fields))
                    // n_bytes_per_sample,
                ),
            ],
        )
        parsed_events.append(
            xr.DataArray(
                name=event_type,
                data=events_.view(dtype=dtype).ravel()["Waveform"],
                dims=("event", "time"),
                coords={
                    dtype.fields[name][2]: (
                        "event",
                        events_.view(dtype=dtype).ravel()[name],
                    )
                    for name in dtype.names
                    if name not in {"reserved", "waveform"}
                },
            ),
        )

    return (
        (
            xr.concat(parsed_events, dim="event")
            .set_xindex(["Packet ID", "Timestamp"])
            .sortby("Timestamp", "Packet ID")
            .rename({"Packet ID": "Electrode ID"})
        )
        if len(parsed_events) > 0
        else None
    )


def _parse_digital_events(
    events: npt.NDArray[np.uint8],
    *,
    packet_size: int,
) -> pd.DataFrame:
    fields = DATA_PACKET_FIELDS["Digital Events"]
    field_name = "Reserved 2"
    dtype = np.dtype(
        [field.to_numpy_dtype() for field in fields]
        + [
            (
                (field_name, field_name.lower().replace(" ", "_")),
                "<B",
                packet_size - sum(field.n_bytes for field in fields),
            ),
        ],
    )
    x = pd.DataFrame(
        {
            dtype.fields[name][2]: events.view(dtype=dtype).ravel()[name]
            for name in dtype.names
            if name not in {"reserved", "packet_id", "reserved_2"}
        },
    )

    return (
        pd.concat(
            [
                x,
                pd.DataFrame(
                    np.unpackbits(
                        np.expand_dims(
                            x["Packet Insertion Reason"],
                            1,
                        ),
                        axis=1,
                    ).astype(bool),
                    columns=(
                        "change to any bit of the digital input parallel port or strobe is triggered",
                        "digital SMA input channel 1 changed",
                        "digital SMA input channel 2 changed",
                        "digital SMA input channel 3 changed",
                        "digital SMA input channel 4 changed",
                        "not used",
                        "periodic sampling event",
                        "serial channel changed",
                    ),
                ).drop(columns=["not used"]),
            ],
            axis=1,
        )
        .set_index("Timestamp")
        .sort_index()
    )


def _parse_diglabel_mode(header: pd.DataFrame) -> pd.DataFrame:
    mode_dtype = pd.CategoricalDtype(
        categories=pd.Index(["serial", "parallel"]),
    )
    return header.assign(
        Mode=lambda x: x["Mode"].replace({0: "serial", 1: "parallel"}),
    ).astype({
        "Mode": mode_dtype,
    })
