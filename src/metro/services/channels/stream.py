from __future__ import annotations

import copy
import traceback
import numpy

from .abstract import AbstractChannel

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, IO


class StreamChannel(AbstractChannel):
    """Robust channel implementation for variable-length numpy arrays.

    This is probably the most commonly used channel throughout the
    various Metro devices. It uses numpy arrays to buffer all provided
    data for a given measurement run. It supports all possbile features
    of AbstractChannel.

    The buffer layout is optimised for frequent insertions rather than
    retrieval. Such an operation may therefore be a rather expensive
    operation (due to the need to compact various data structures) and
    should be avoided during a measurement run. Consider blocking
    between steps in this case.

    In addition to the parameters of AbstractChannel, a shape parameter
    should be provided to determine the dimensionality of the data in
    this channel, which defines the data type for the dataAdded
    callback.

        0: scalar data, addData/dataAdded uses int
        1: numpy array of shape(X,) with X being an arbitrary sample
            amount, addData/dataAdded uses array(X,)
        N(>1): numpy array of shape(X,N) with X being an arbitrary
            sample amount, addData/dataAdded uses array(X,N)

    There is also the option to provide an interval [a,b] via
    setInterval() that guarentees that for all samples x in this
    channel: a <= x <= b. Subscribers may use the rangeChanged callback
    to be notified of any changes to this interval. This range has to
    provided manually by the channel provider using setRange()!

    Parameters
    ----------
    *names
        see AbstractChannel
    shape: int
        A non-negative integer describing the dimensionality
        of data contained in this channel. Please see the
        general class documentation for details.
    **options
        see AbstractChannel

    """

    counter_func_scalar = staticmethod(lambda x: 1)
    emptySet_func_scalar = staticmethod(lambda x: False)

    counter_func_vector = len
    emptySet_func_vector = staticmethod(lambda x: x.size == 0)

    def __init__(self, *names, shape: int = 0, **options) -> None:
        self.data = [[]]

        self.locked = False

        try:
            self.buffering = bool(options["buffering"])

        except KeyError:
            self.buffering = True

        try:
            self.transient = bool(options["transient"])

        except KeyError:
            self.transient = False

        self.current_index = 0
        self.step_values = []

        self.shape = int(shape)

        if self.shape == 0:
            self.counter_func = self.counter_func_scalar
            self.emptySet_func = self.emptySet_func_scalar

            self._writeData = self._writeData_scalar
            self._compactData = self._compactData_scalar

        elif self.shape > 0:
            self.counter_func = self.counter_func_vector
            self.emptySet_func = self.emptySet_func_vector

            self._writeData = self._writeData_vector
            self._compactData = self._compactData_vector

        else:
            raise ValueError("shape must be non-negative integer value")

        self.range_min = None
        self.range_max = None

        super().__init__(*names, **options)

        self.header_tags["Shape"] = self.shape

    # PRIVATE METHODS
    @staticmethod
    def _writeData_scalar(fp: IO, d: Any) -> None:
        """Write channel data to a file pointer."""

        if isinstance(d, numpy.ndarray):
            numpy.savetxt(fp, d, delimiter="\t")
        else:
            fp.write("{0}\n".format(d).encode("utf-8"))

    @staticmethod
    def _compactData_scalar(data: Any) -> List[numpy.ndarray]:
        """Compact channel data."""

        if len(data) == 0:
            return data

        if not isinstance(data[0], numpy.ndarray):
            # The complete step is still a python list, simply wrap a
            # numpy array around it. We optimize this special case since
            # no concatenation is needed.
            return [numpy.array(data)]

        else:
            # Here the first element is already a numpy array, so we
            # compacted once in this step. Wrap the remaining elements
            # in a new array and concatenate it.

            old_array = data[0]
            new_array = numpy.array(data[1:])

            return [numpy.concatenate([old_array, new_array])]

    @staticmethod
    def _writeData_vector(fp: IO, d: Any) -> None:
        """Write channel data to a file pointer."""

        numpy.savetxt(fp, d, delimiter="\t")

    @staticmethod
    def _compactData_vector(data: Any) -> None:
        """Compact channel data."""

        return [numpy.concatenate(data)]

    @staticmethod
    def _printException(e: Exception) -> None:
        """Pretty-print an exception."""

        print(
            "An exception was raised by a channel subscriber, which may "
            "cause other subscribers of the same channel to miss this "
            "callback. The data is still saved in the channel buffers (and "
            "written to disk in storage mode.\nThe causing exception reads:"
        )

        traceback.print_exception(type(e), e, e.__traceback__)

    def addMarker(self, text: str) -> None:
        """Write a custom marker.

        Args;
            text: A string containing the marker to write.
        """

        try:
            fp = self.file_pointer
        except AttributeError:
            pass
        else:
            fp.write("# {0}\n".format(text).encode("ascii"))

    # PUBLIC IMPLEMENTATION API
    def beginScan(self, scan_counter: int) -> None:
        """Begin a scan."""

        super().beginScan(scan_counter)

        self.addMarker("SCAN {0}".format(scan_counter))

        # Gets incremented by beginStep to 0
        self.current_index = -1

    def beginStep(self, step_value: Any) -> None:
        """Begin a step."""

        try:
            self.data_file_pointer = self.file_pointer
        except AttributeError:
            pass

        super().beginStep(step_value)

        self.current_index += 1

        if step_value is None:
            step_value = str(self.current_index)

        try:
            self.step_values[self.current_index] = step_value
        except IndexError:
            self.step_values.append(step_value)

        if self.current_index > len(self.data) - 1 and self.buffering:
            self.data.append([])

        if self.freq == AbstractChannel.CONTINUOUS_SAMPLES:
            self.addMarker("STEP {0}: {1}".format(self.current_index, step_value))

            for s in self.subscribers:
                step_index = s._channel_subscriber_step_index

                if (
                    step_index == AbstractChannel.CURRENT_STEP
                    or step_index == self.current_index
                ):
                    if self.buffering and len(self.data[self.current_index]) > 0:
                        s.dataSet(self.getData())
                    else:
                        s.dataCleared()
        elif self.freq == AbstractChannel.SCHEDULED_SAMPLES:
            self.addMarker("STEP {0}: {1}".format(self.current_index, step_value))

    def endStep(self) -> None:
        """End a step."""

        super().endStep()

        try:
            del self.data_file_pointer
        except AttributeError:
            pass

        try:
            self.file_pointer.flush()
        except AttributeError:
            pass
        except ValueError as e:
            print("ValueError on flush of", self.name, e)

    def copyDataFrom(self, chan: AbstractChannel) -> None:
        """Copy the data from another into this channel.

        Args:
            chan: Channel to copy the data from.
        """

        self.data = copy.deepcopy(chan.data)
        self.step_values = copy.deepcopy(chan.step_values)

        self.current_index = chan.current_index

    def openStorage(self, base_path: str) -> None:
        if self.transient:
            return

        # If a file pointer exists, we are already storing
        if hasattr(self, "file_pointer"):
            return

        # The file is opened in binary mode since apparently
        # numpy.savetxt operates on byte buffers instead of strings. We
        # therefore have to encode all our own strings ourselves!
        fp = open("{0}_{1}.txt".format(base_path, self.name), "wb")

        fp.write(
            "# Name: {0}\n# Hint: {1}\n# Frequency: {2}\n".format(
                self.name,
                self.getHintString(self.hint),
                self.getFrequencyString(self.freq),
            ).encode("ascii")
        )

        for tag, value in self.header_tags.items():
            fp.write("# {0}: {1}\n".format(tag, value).encode("ascii"))

        for key, value in self.display_arguments.items():
            fp.write("# DISPLAY {0}: {1}\n".format(key, value).encode("ascii"))

        self.file_pointer = fp

    def closeStorage(self) -> None:
        try:
            fp = self.file_pointer
        except AttributeError:
            pass
        else:
            fp.close()
            del self.file_pointer

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

        if not silent and self.buffering:
            d = self.getData()

            if d is None:
                obj.dataCleared()
            else:
                obj.dataSet(d)

    def setSubscribedStep(self, obj: Subscriber, step_index: int) -> None:
        """Set the subscribed step for an object.

        Changing the subscribed step for a subscriber may trigger a
        dataSet callback if the new step contains data.

        Args:
            obj: The subscriber object the step should be changed for.
            step_index: An integer containing either a specific step or
                a special index.
        """

        super().setSubscribedStep(obj, step_index)

        if not self.buffering:
            return

        try:
            data = self.getData(step_index)
        except ValueError:
            pass
        else:
            obj.dataSet(data)

    def reset(self) -> None:
        """Reset the channel."""

        self.data.clear()

        self.data.append([])
        self.current_index = 0

        for s in self.subscribers:
            s.dataCleared()

    def isEmpty(self) -> bool:
        """Check if the active step is empty."""

        if not self.buffering:
            return True

        # UNUSED METHOD?!

        return len(self.data[self.current_index]) == 0

    def getStepCount(self) -> int:
        """Get the number of steps in this channel's buffers."""

        return len(self.data)

    def copyLayoutFrom(self, ch: AbstractChannel) -> None:
        """Copy step layout of another channel."""

        if isinstance(ch, NumericChannel):
            self.step_values = ch.step_values

        step_diff = ch.getStepCount() - len(self.data)

        if step_diff > 0:
            for i in range(step_diff):
                self.data.append([])

    # PUBLIC USER API
    def dump(self, step: int = AbstractChannel.CURRENT_STEP, fp: IO = None) -> None:
        """Dump channel data."""

        if fp is None:
            try:
                fp = self.file_pointer
            except AttributeError:
                raise ValueError(
                    "no file pointer supplied and channel is not in storage mode"
                )

        d = self.getData(step)

        if d is not None:
            self._writeData(fp, d)

    def setAveraging(self, ch_input: AbstractChannel) -> None:
        self.setIntegrating(numpy.mean, [ch_input])

    def setAccumulating(self, ch_input: AbstractChannel) -> None:
        self.setIntegrating(numpy.sum, [ch_input])

    def getData(self, step_index: int = AbstractChannel.CURRENT_STEP) -> numpy.ndarray:
        """Get channel data."""

        if not self.buffering:
            return None

        if step_index == AbstractChannel.CURRENT_STEP:
            step_index = self.current_index

        # TODO: We could optimize the data layout for non-CONTINUOUS
        # channels to use "less arrays" and not having to create new
        # ones in this method.
        if self.freq == AbstractChannel.CONTINUOUS_SAMPLES and step_index > -1:
            try:
                step = self.data[step_index]
            except IndexError:
                raise ValueError("step index out of range")

            step_len = len(step)

            if step_len == 0:
                return None
            elif step_len > 0:
                self.data[step_index] = self._compactData(step)
                step = self.data[step_index]

            return step[0]

        else:
            # Every frequency except CONTINUOUS_SAMPLES implies
            # step_index == ALL_STEPS

            buf = []

            i = 0
            for step in self.data:
                if len(step) > 0:
                    if self.freq == AbstractChannel.STEP_SAMPLES:
                        # No compacting necessary, we always take the last
                        # sample (i.e. the last performed scan).
                        buf.append(step[-1])
                    else:
                        new_step = self._compactData(step)
                        buf.append(new_step[0])

                        self.data[i] = new_step

                i += 1

            if len(buf) == 0:
                return None

            if self.freq == AbstractChannel.STEP_SAMPLES:
                return numpy.array(buf)
            else:
                return numpy.concatenate(buf)

    def setData(self, d: Any, step_index: int = AbstractChannel.CURRENT_STEP) -> None:
        """Set channel data."""

        # Only necessary for scalar data
        if not isinstance(d, numpy.ndarray):
            d = numpy.array([d])

        if d is None or self.emptySet_func(d):
            self.clearData()
            return

        if step_index == AbstractChannel.CURRENT_STEP:
            step_index = self.current_index
        elif step_index >= len(self.data):
            raise ValueError("step index out of range")

        self.data[step_index] = [d]

        for s in self.subscribers:
            subscribed_index = s._channel_subscriber_step_index

            if (
                subscribed_index == AbstractChannel.CURRENT_STEP
                and step_index == self.current_index
            ):
                s.dataSet(d)
            elif subscribed_index == step_index:
                s.dataSet(d)
            elif subscribed_index == AbstractChannel.ALL_STEPS:
                s.dataSet(self.getData(AbstractChannel.ALL_STEPS))

    def addData(self, d: Any) -> None:
        """Add channel data."""

        if d is None or self.emptySet_func(d):
            return

        if self.buffering:
            self.data[self.current_index].append(d)

        for s in self.subscribers:
            step_index = s._channel_subscriber_step_index

            if step_index < 0 or step_index == self.current_index:
                s.dataAdded(d)

        try:
            fp = self.data_file_pointer
        except AttributeError:
            pass
        else:
            self._writeData(fp, d)

    def clearData(self) -> None:
        """Clear channel data."""

        self.data[self.current_index].clear()

        for s in self.subscribers:
            step_index = s._channel_subscriber_step_index

            if (
                step_index == AbstractChannel.CURRENT_STEP
                or step_index == self.current_index
            ):
                s.dataCleared()

    def isBuffering(self) -> bool:
        """Returns whether this channel is buffering.

        Returns:
            A boolean indicating the buffering state
        """

        return self.buffering

    def getRange(self) -> Tuple[float, float]:
        """Return the range of values of this channel.

        Returns:
            A tuple of floats in the form (min, max)
        """

        return self.range_min, self.range_max

    def setRange(self, new_min: float, new_max: float) -> None:
        """Set the range of values in this channel.

        Args:
            range_min: A float that is lower or equal than all other
                samples in this channel.
            range_max: A float that is greater or equal than all other
                samples in this channel.
        """

        self.range_min = new_min
        self.range_max = new_max

        self._notify("rangeChanged")


NumericChannel = StreamChannel  # previous name for compatibility
