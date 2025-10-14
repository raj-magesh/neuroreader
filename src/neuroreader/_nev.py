import functools
import os
from pathlib import Path
from typing import TypedDict, cast, final

import numpy as np
import numpy.typing as npt
import pandas as pd
import pint_pandas  # noqa: F401
import xarray as xr

from neuroreader._utilities import (
    DataPacket,
    Field,
    Header,
    parse_file_spec,
    parse_filter_details,
    parse_time_origin,
    parse_time_resolution,
    read_field,
    read_fields,
)

N_BYTES_INT16 = 2
N_BYTES_INT32 = 4

Events = TypedDict(
    "Events",
    {
        "Digital Events": pd.DataFrame | None,
        "Spike Events": xr.DataArray | None,
        "Stimulation Events": xr.DataArray | None,
    },
)

DigitalEventDataPacket = TypedDict(
    "DigitalEventDataPacket",
    {
        "Timestamp": int,
        "Packet ID": int,
        "Packet Insertion Reason": int,
        "Reserved": None,
        "Parallel Input": int,
        "SMA Input 1": int,
        "SMA Input 2": int,
        "SMA Input 3": int,
        "SMA Input 4": int,
    },
)

SpikeEventDataPacket = TypedDict(
    "SpikeEventDataPacket",
    {
        "Timestamp": int,
        "Packet ID": int,
        "Unit Classification Number": int,
        "Reserved": None,
        "Waveform": npt.NDArray[np.integer],
    },
)

StimulationEventDataPacket = TypedDict(
    "StimulationEventDataPacket",
    {
        "Timestamp": int,
        "Packet ID": int,
        "Reserved": None,
        "Waveform": npt.NDArray[np.integer],
    },
)

EventDataPacket = (
    DigitalEventDataPacket | SpikeEventDataPacket | StimulationEventDataPacket
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

DATA_PACKET_FIELDS: dict[frozenset[int], list[Field]] = {
    frozenset([0]): [
        Field("Timestamp", "<", "I", 4),
        Field("Packet ID", "<", "H", 2),
        Field("Packet Insertion Reason", "<", "B", 1),
        Field("Reserved", "<", "B", 1),
        Field("Parallel Input", "<", "H", 2),
        Field("SMA Input 1", "<", "h", 2),
        Field("SMA Input 2", "<", "h", 2),
        Field("SMA Input 3", "<", "h", 2),
        Field("SMA Input 4", "<", "h", 2),
    ],
    frozenset(range(1, 512 + 1)): [
        Field("Timestamp", "<", "I", 4),
        Field("Packet ID", "<", "H", 2),
        Field("Unit Classification Number", "<", "B", 1),
        Field("Reserved", "<", "B", 1),
    ],
    frozenset(range(5121, 5632 + 1)): [
        Field("Timestamp", "<", "I", 4),
        Field("Packet ID", "<", "H", 2),
        Field("Reserved", "<", "B", 2),
    ],
}


@final
class NEV:
    def __init__(self, filepath: Path) -> None:
        self._filepath = filepath
        self._read_headers()

    @functools.cached_property
    def basic_header(self) -> Header:
        return self._basic_header

    @functools.cached_property
    def extended_headers(self) -> dict[str, pd.DataFrame]:
        return self._extended_headers

    @functools.cached_property
    def data(self) -> xr.DataArray:
        self._read_data_packets()
        return self._data

    @property
    def spikes(self) -> xr.DataArray | None:
        return self.data["Spike Events"]

    @property
    def stimulations(self) -> xr.DataArray | None:
        return self.data["Stimulation Events"]

    @property
    def digital_events(self) -> xr.DataArray | None:
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
        fields = {
            "timestamp": Field("", "<", "I", 4),
            "packet_id": Field("", "<", "H", 2),
        }

        bytes_per_sample = self.extended_headers["NEUEVWAV"][
            "Bytes per Sample"
        ].to_dict()
        packet_size = cast("int", self.basic_header["Bytes in Data Packets"])

        data_packets: list[DataPacket] = []

        n_bytes_in_headers = cast("int", self.basic_header["Bytes in Headers"])

        with self._filepath.open("rb") as f:
            file_size = f.seek(0, os.SEEK_END)
            _ = f.seek(n_bytes_in_headers, os.SEEK_SET)

            while f.tell() < file_size:
                timestamp = cast("int", read_field(f, field=fields["timestamp"]))

                if hex(timestamp) == "0xffffffff":
                    # TODO
                    raise NotImplementedError

                packet_id = cast("int", read_field(f, field=fields["packet_id"]))
                _ = f.seek(
                    -(fields["timestamp"].n_bytes + fields["packet_id"].n_bytes),
                    os.SEEK_CUR,
                )

                packet_id_identified = False
                for key in DATA_PACKET_FIELDS:
                    if packet_id in key:
                        packet_id_identified = True
                        break

                if not packet_id_identified:
                    break

                packet = read_fields(f, fields=DATA_PACKET_FIELDS[key])

                if packet_id == 0:
                    packet["Reserved"] = read_field(
                        f,
                        field=Field("", "<", "s", packet_size - 18),
                    )
                else:
                    n_bytes = bytes_per_sample[packet["Packet ID"]]

                    if n_bytes == N_BYTES_INT16:
                        dtype = np.dtype(np.int16)
                    elif n_bytes == N_BYTES_INT32:
                        dtype = np.dtype(np.int32)
                    else:
                        error = f"`Bytes per Sample` for `Electrode ID` {packet['Electrode ID']} is {n_bytes}, but only 2- and 4-byte integers are supported"
                        raise ValueError(error)

                    shape: float = (packet_size - 8) / n_bytes
                    if shape.is_integer():
                        shape = int(shape)
                    else:
                        raise ValueError

                    packet["Waveform"] = np.memmap(
                        self._filepath,
                        dtype=dtype.newbyteorder("<"),
                        mode="r",
                        offset=f.tell(),
                        shape=(shape,),
                    )
                    _ = f.seek(packet_size - 8, os.SEEK_CUR)

                data_packets.append(packet)

        self._data = _parse_data_packets(data_packets)


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
            "Packet ID": "string",
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
            "Packet ID": "string",
            "Electrode ID": np.uint16,
            "Reserved": np.bytes_,
        })
        .set_index("Electrode ID")
        .drop(columns=["Packet ID", "Reserved"])
    )

    headers["NEUEVLBL"] = (
        headers["NEUEVLBL"]
        .astype({
            "Packet ID": "string",
            "Electrode ID": np.uint16,
            "Label": "string",
            "Reserved": np.bytes_,
        })
        .set_index("Electrode ID")
        .drop(columns=["Packet ID", "Reserved"])
    )

    headers["DIGLABEL"] = (
        _parse_diglabel_mode(headers["DIGLABEL"])
        .astype({
            "Label": "string",
        })
        .drop(columns=["Packet ID", "Reserved"])
        .set_index("Label")
    )

    return headers


def _parse_data_packets(
    data_packets: list[EventDataPacket],
) -> Events:
    data: dict[int, list[EventDataPacket]] = {}

    for packet in data_packets:
        packet_id = packet["Packet ID"]

        if packet_id in data:
            data[packet_id].append(packet)
        else:
            data[packet_id] = [packet]

    events: Events = {
        "Digital Events": None,
        "Spike Events": None,
        "Stimulation Events": None,
    }
    events_: dict[str, list[xr.DataArray]] = {
        "Spike Events": [],
        "Stimulation Events": [],
    }

    for packet_id, data_ in data.items():
        event = pd.DataFrame(data_).drop(columns=["Packet ID", "Reserved"])

        types = {"Timestamp": np.uint32}

        if packet_id == 0:
            types |= {
                "Parallel Input": np.uint16,
                "Packet Insertion Reason": np.uint8,
            }
            types |= {f"SMA Input {1 + idx}": np.int16 for idx in range(4)}
        elif packet_id in set(range(1, 512 + 1)):
            types |= {"Unit Classification Number": np.uint8}
            event_type = "Spike Events"
        elif packet_id in set(range(5121, 5632 + 1)):
            event_type = "Stimulation Events"
        else:
            raise ValueError

        event = event.astype(types)

        if (packet_id == 0) and (events["Digital Events"] is None):
            events["Digital Events"] = _parse_digital_events(event)
            continue

        events_[event_type].append(
            xr.DataArray(
                name=event_type,
                data=np.stack(event["Waveform"]),
                dims=("event", "time"),
                coords={
                    column: ("event", coord)
                    for column, coord in event.drop(columns=["Waveform"]).items()
                }
                | {
                    "Electrode ID": (
                        "event",
                        packet_id * np.ones((len(event),), dtype=np.uint16),
                    ),
                },
            ),
        )

    for event_type in ("Spike Events", "Stimulation Events"):
        if len(events_[event_type]) > 0:
            events[event_type] = (
                xr.concat(events_[event_type], dim="event")
                .set_xindex(["Electrode ID", "Timestamp"])
                .sortby("Electrode ID", "Timestamp")
            )
        else:
            events[event_type] = None

    return events


def _parse_digital_events(x: pd.DataFrame) -> pd.DataFrame:
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
