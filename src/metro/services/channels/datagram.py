from __future__ import annotations

import numpy
import h5py

from . import Frequency, Step
from .abstract import AbstractChannel
from .subscriber import Subscriber

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


class DatagramChannel(AbstractChannel):
    def __init__(
        self,
        *names,
        compression: bool | int = False,
        transient: bool = False,
        **options,
    ) -> None:
        if compression:
            self.compress_args = {
                "compression": "gzip",
                "compression_opts": (
                    4 if isinstance(compression, bool) else int(compression)
                ),
            }

        else:
            self.compress_args = {}

        self.transient = transient

        self.image_idx = 0

        self.storage_base = None
        self.next_dset_name = None

        self.last_datum = None
        self.last_metadata = None

        super().__init__(*names, **options)

    def _addMetaData(self) -> None:
        attrs = self.h5file.attrs

        attrs["name"] = self.name
        attrs["freq"] = self.freq.name.lower()
        attrs["hint"] = self.hint.name.lower()

        for tag, value in self.header_tags.items():
            attrs[tag] = value

        for key, value in self.display_arguments.items():
            attrs["DISPLAY " + key] = value

    def openStorage(self, base_path: str) -> None:
        if self.transient:
            return

        self.storage_base = base_path

        if self.freq is Frequency.STEP:
            self.h5file = h5py.File(
                "{0}_{1}.h5".format(self.storage_base, self.name), "w"
            )

    def closeStorage(self) -> None:
        if self.storage_base is not None and self.freq is Frequency.STEP:
            self._addMetaData()
            self.h5file.close()
            del self.h5file

        self.storage_base = None

    def subscribe(self, obj: Subscriber, silent: bool = False) -> None:
        """Subscribe to this channel.

        Add a subscriber object to this channel that receives callbacks.

        Args:
            obj: The Subscriber object to be added
            silent: Optional boolean to indicate that no callbacks
                should be fired upon subscribing. This may include the
                added or cleared callback depending on the channel's
                data content.
        """

        super().subscribe(obj)

        if not silent:
            if self.last_datum is None:
                obj.dataCleared()

            else:
                obj.dataAdded(self.last_datum)

    def beginScan(self, scan_counter: int) -> None:
        super().beginScan(scan_counter)

        if self.storage_base is not None and self.freq is Frequency.STEP:
            self.h5scan = self.h5file.create_group(str(scan_counter))

        self.step_idx = -1

    def beginStep(self, step_value: float) -> None:
        super().beginStep(step_value)

        self.step_idx += 1
        self.image_idx = 0

        if self.storage_base is not None:
            if self.freq is Frequency.CONTINUOUS:
                self.h5file = h5py.File(
                    "{0}_{1}_{2}.h5".format(
                        self.storage_base, self.name, self.step_idx
                    ),
                    "w",
                )

            elif self.freq is Frequency.STEP:
                self.next_dset_name = str(step_value)

    def endStep(self) -> None:
        if self.storage_base is not None:
            if self.freq is Frequency.CONTINUOUS:
                self._addMetaData()
                self.h5file.close()
                del self.h5file

            elif self.freq is Frequency.STEP:
                if self.last_datum is None:
                    return

                im_name = (
                    self.next_dset_name
                    if self.next_dset_name is not None
                    else str(self.image_idx)
                )

                im_dset = self.h5scan.create_dataset(
                    im_name,
                    data=self.last_datum,
                    chunks=self.last_datum.shape,
                    **self.compress_args,
                )

                self.last_datum = None

                if self.last_metadata is None:
                    return

                for key, value in self.last_metadata.items():
                    im_dset.attrs[key] = value

                self.last_metadata = None

    def reset(self) -> None:
        for s in self.subscribers:
            s.dataCleared()

    def getData(self, step_index: Step = Step.CURRENT) -> numpy.ndarray:
        return self.last_datum

    def setData(self, d: Any) -> None:
        raise NotImplementedError("setData not supported by DatagramChannel (yet!)")

    def addData(self, d: Any, **metadata: Any):
        if self.storage_base is not None and self.freq is Frequency.CONTINUOUS:
            im_name = (
                self.next_dset_name
                if self.next_dset_name is not None
                else str(self.image_idx)
            )

            im_dset = self.h5file.create_dataset(
                im_name, data=d, chunks=d.shape, **self.compress_args
            )

            for key, value in metadata.items():
                im_dset.attrs[key] = value

        for s in self.subscribers:
            step_index = s._channel_subscriber_step_index

            if step_index < 0 or step_index == self.current_index:
                s.dataAdded(d)

        self.last_datum = d
        self.last_metadata = metadata

        self.image_idx += 1
