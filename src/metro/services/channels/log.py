from __future__ import annotations

import datetime
import multiprocessing
from time import time as time_now
import numpy
import h5py

from .abstract import AbstractChannel

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


class LogChannel(AbstractChannel):
    OP_ADD_DATA = 0
    OP_OPEN_CHANNEL = 1
    OP_CLOSE_CHANNEL = 2
    OP_QUIT = 3

    storage_root = "."
    ch_count = 0
    logger_p = None
    op_in = None
    compression_args = dict(compression="gzip", compression_opts=4)

    def __init__(
        self,
        *names,
        interval: int = -1,
        fields: List[Tuple[str, str]] = [],
        **options,
    ) -> None:
        if interval <= 0:
            raise ValueError("interval must be greater than zero")

        if fields is None:
            raise ValueError("no fields to log")

        options["static"] = True

        super().__init__(*names, **options)

        if LogChannel.ch_count == 0:
            op_out, op_in = multiprocessing.Pipe(False)

            LogChannel.logger_p = multiprocessing.Process(
                target=LogChannel.logger_main,
                args=(LogChannel.storage_root, op_out),
            )
            LogChannel.logger_p.start()

            LogChannel.op_in = op_in

        LogChannel.op_in.send((LogChannel.OP_OPEN_CHANNEL, self.name, interval, fields))
        LogChannel.ch_count += 1

    def close(self) -> None:
        super().close()

        LogChannel.op_in.send((LogChannel.OP_CLOSE_CHANNEL, self.name))
        LogChannel.ch_count -= 1

        if LogChannel.ch_count == 0:
            LogChannel.op_in.send((LogChannel.OP_QUIT,))
            LogChannel.logger_p.join()

            LogChannel.logger_p = None
            LogChannel.op_in = None

    @staticmethod
    def logger_main(storage_root, pipe) -> None:
        # h5f, flush_size, dtype, empty_rec
        cur_channels = {}

        while pipe.poll(None):
            msg = pipe.recv()

            if msg[0] == LogChannel.OP_ADD_DATA:
                _, name, label, time, data = msg

                time = datetime.datetime.fromtimestamp(time)
                dset_name = "{0}/{1}".format(label, time.strftime("%Y-%m-%d"))

                h5f, interval, dtype, empty_rec = cur_channels[name]
                h5d = None

                if dset_name in h5f:
                    h5d = h5f[dset_name]

                    if h5d.dtype != dtype:
                        # If we have a dtype mismatch, we rename the
                        # "wrong" dataset and create our own new one.

                        replacement_name = dset_name + "_DTYPE_MISMATCH"
                        i = 2

                        while replacement_name in h5f:
                            replacement_name = "{0}_DTYPE_MISMATCH{1}".format(
                                dset_name, i
                            )
                            i += 1

                        h5f.create_dataset(
                            replacement_name,
                            dtype=h5d.dtype,
                            data=h5d,
                            **LogChannel.compression_args,
                        )
                        del h5f[dset_name]
                        h5d = None

                if h5d is None:
                    h5d = h5f.create_dataset(
                        dset_name,
                        shape=(1,),
                        maxshape=(None,),
                        chunks=(int(3600 / interval),),
                        dtype=dtype,
                        **LogChannel.compression_args,
                    )
                else:
                    h5d.resize(h5d.shape[0] + 1, axis=0)

                empty_rec[0][0] = time.strftime("%H%M%S").encode("ascii")
                for i in range(len(data)):
                    empty_rec[0][i + 1] = data[i]

                h5d[-1] = empty_rec

                if (h5d.shape[0] % max(int(300 / interval), 1)) == 0:
                    h5f.flush()

            elif msg[0] == LogChannel.OP_OPEN_CHANNEL:
                _, channel_name, interval, fields = msg

                if channel_name in cur_channels:
                    continue

                fields.insert(0, ("time", "S6"))
                dtype = numpy.dtype(fields, align=False)
                empty_rec = numpy.empty((1,), dtype)

                h5f = h5py.File("{0}/{1}.h5".format(storage_root, channel_name), "a")

                cur_channels[channel_name] = (h5f, interval, dtype, empty_rec)

            elif msg[0] == LogChannel.OP_CLOSE_CHANNEL:
                channel_name = msg[1]

                if channel_name not in cur_channels:
                    continue

                cur_channels[channel_name][0].close()
                del cur_channels[channel_name]

            elif msg[0] == LogChannel.OP_QUIT:
                break

        for ch_entry in cur_channels.values():
            ch_entry[0].close()

        # Exit either after 60s passed without any message or we got an
        # explicit command for it

    def openStorage(self, base_path: str) -> None:
        raise NotImplementedError("non-static storage not supported by LogChannel")

    def closeStorage(self) -> None:
        raise NotImplementedError("non-static storage not supported by LogChannel")

    def setData(self, d: Any) -> None:
        raise NotImplementedError("setData not supported by LogChannel")

    def addData(self, *d, label: str = "", time: int = 0) -> None:
        if time == 0:
            time = time_now()

        LogChannel.op_in.send((LogChannel.OP_ADD_DATA, self.name, label, time, d))

    def clearData(self) -> None:
        raise NotImplementedError("clearData not supported by LogChannel")


# For compatibility with older versions of Python on Windows (i.e. using
# the spawn multiprocessing method), which look for 'logger_main' in the
# module namespace
logger_main = LogChannel.logger_main
