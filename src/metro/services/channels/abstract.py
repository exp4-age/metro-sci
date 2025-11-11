from __future__ import annotations

import functools
import traceback

from . import Mode, Hint, Frequency, Step
from .subscriber import Subscriber

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, IO, Mapping


class AbstractChannel(object):
    """Abstract base class for all channels.

    This class defines the general interface and functionality common to
    all channels. It contains an implementation for subscribing/
    unsubschribing from a channel as well as handling direct/remote/
    computing/integrating mode. It is still completely independant of
    any data model for the actual data the channel will contain in the
    end.

    A new channel implementation can therefore either extend this class
    and implement this completely on its own or extend LocalChannel,
    which uses python lists and numpy arrays to hold the data.

    All methods in this class are NOT thread-safe and may NOT be called
    from any other thread but the main thread! This is a hard assumption
    for all code in this class and any extending class, since channels
    make heavy use of direct callbacks for performance reasons and have
    to guarantee to this callback code to also run on the main thread.
    Again, calling it from other threads WILL break things!

    This abstract class is incomplete and requires the implementation
    of several methods that will raise NotImplementedError by default:

        reset
        isEmpty
        getStepCount

        getData
        setData
        addData
        clearData

    All other methods that are overriden should call the super
    implementation as well.

    Parameters
    ----------
    *names
        All positional arguments are converted into strings
        and concatenated with the '#' into the name of this channel.
    **options
        The keyword parameters are used to specify various channel
        parameters: hint, freq and static (a boolean to indicate whether
        this channel is static. Such channels do not participate in
        measurements and do not change their data contents.).

    """

    def __init__(
        self,
        *names,
        hint: Hint = Hint.WAVEFORM,
        freq: Frequency = Frequency.CONTINUOUS,
        static: bool = False,
    ) -> None:
        name = "#".join([str(x) for x in names])

        if name in _channels:
            raise ValueError('name "{0}" already in use'.format(name))

        self.name = name

        self.locked = False
        self.subscribers = []
        self.listener = []
        self.header_tags = {}
        self.display_arguments = {}

        # Options
        self.mode = Mode.DIRECT
        self.hint = hint
        self.freq = freq
        self.static = static

        _channels[self.name] = self

        self._notify("channelOpened")

    def __str__(self) -> None:
        return "{0}({1})".format(self.__class__.__name__, self.name)

    # PRIVATE METHODS
    def _notify(self, callback: str) -> None:
        """Trigger a callback to channel watchers.

        This method will call a method with the supplied name on all
        channel watchers whose watch parameters are compatible with the
        name of this channel as the only argument.

        Parameters
        ----------
        callback: str
            String with the method name to call on watchers

        """

        for watcher in _watchers:
            opts = _watchers[watcher]

            if opts[0] is not None and opts[0] is not self.hint:
                continue

            elif opts[1] is not None and opts[1] is not self.freq:
                continue

            elif opts[2] is not None and not isinstance(self, opts[3]):
                continue

            elif opts[3] is not None:
                try:
                    if opts[3] != self.shape:
                        continue

                except AttributeError:
                    continue

            elif opts[4] is not None and opts[4] != self:
                continue

            elif opts[5] is not None and callback not in opts[5]:
                continue

            try:
                method = getattr(watcher, callback)

            except AttributeError:
                pass

            else:
                method(self)

    def _computing_single_dataAdded(self, d: Any) -> None:
        """Channel callback in computing mode.

        Callback for dataAdded on the input channel in computing mode.
        Note that this version is only used if the kernel argument
        consists of exactly one argument as an optimized form of
        _computing_multiple_dataAdded.

        """

        try:
            value = self.kernel(d)

        except Exception as e:
            print(
                "An unechecked exception was raised in the computing "
                "kernel of {0}:".format(self.name)
            )

            traceback.print_exception(type(e), e, e.__traceback__)

        else:
            self.addData(value)

    def _computing_multiple_dataAdded(self, d: Any, index: int) -> None:
        """Channel callback in computing mode.

        Callback for dataAdded on the input channels in computing mode.
        It has been modified with functools to always include an index
        identifying the channel adding data.  Note that this is only
        used if the kernel argument consists of more than one channel.

        """

        self.input_stack[index] = d

        if None not in self.input_stack:
            try:
                value = self.kernel(*self.input_stack)

            except Exception as e:
                print(
                    "An unechecked exception was raised in the computing "
                    "kernel of {0}:".format(self.name)
                )

                traceback.print_exception(type(e), e, e.__traceback__)

            else:
                self.addData(value)

            self.input_stack = [None] * len(self.input_channels)

    def _stopComputing(self) -> None:
        """Terminate computing mode.

        The actual mode flag is not changed, but the used resources are
        properly deallocated such as channel subscriptions.

        """

        for subscr in self.input_subscriber:
            subscr.channel.unsubscribe(subscr)

        self.kernel = None
        self.input_channels = None
        self.input_subscriber = None
        self.input_stack = None

    def _stopIntegrating(self) -> None:
        """Terminate integrating mode.

        The actual mode flag is not changed, but the used resources are
        properly deallocated.

        """

        self.kernel = None
        self.input_channels = None

    # PUBLIC IMPLEMENTATION API
    def dependsOn(self, channel: AbstractChannel) -> bool:
        """Check for dependance on other channel.

        A channel might depend on another channel such as in computing
        mode it depends on its input channels to properly emit all
        samples before ending a step. This method is used by
        sortByDependency to sort a list of channels according to these
        dependencies.

        Parameters
        ----------
        channel: AbstractChannel
            Channel object to test dependency for (whether this
            channel depends on the input channel).

        Returns
        -------
        bool
            A boolean indicating dependency.

        """

        try:
            return channel in self.input_channels

        except AttributeError:
            return False

    def isStatic(self) -> bool:
        """Check if the channel is static.

        A static channel will be ignored by the measuring process. It
        may be used to store static data independant of measurements

        Returns
        -------
        bool
            A boolean indicating whether the channel is static.

        """

        return self.static

    def beginScan(self, scan_counter: int) -> None:
        """Begin a scan.

        The measuring controller calls this method at the begin of every
        scan iteration, so that a channel can properly prepare. One
        consequence in AbstractChannel is the lockdown of this channel
        to measuring mode, which prohibits certain operations.

        Any data operation before beginScan and after the respective
        endScan should be considered offline.

        Parameters
        ----------
        scan_counter: int
            The current scan iteration counter.

        """

        self.locked = True

    def beginStep(self, step_value: Any) -> None:
        """Begin a step.

        The measuring controller calls this method at the begin of every
        step, so that a channel can properly prepare.

        Parameters
        ----------
        step_value: Any
            A value describing this step of arbitrary type,
            which is assumed to stay constant for the same step
            across several scan iterations.

        """

        pass

    def endStep(self) -> None:
        """End a step.

        The measuring controller calls this method at the end of every
        step, so that a channel can perform cleanup work.

        In integrating mode, the result for a step is calculated in this
        method.

        """

        if self.mode is Mode.INTEGRATING:
            stack = []

            for ch in self.input_channels:
                d = ch.getData()

                if d is None:
                    stack = None
                    break

                stack.append(ch.getData())

            if stack is not None:
                try:
                    value = self.kernel(*stack)

                except Exception as e:
                    print(
                        "An unechecked exception was raised in the "
                        "integrating kernel of {0}:".format(self.name)
                    )

                    traceback.print_exception(type(e), e, e.__traceback__)

                else:
                    self.addData(value)

    def endScan(self) -> None:
        """End a scan.

        The measuring controller calls this method at the end of every
        scan iteration, so that a channel can perform cleanup work.

        Any data operation after endScan and before the next beginScan
        should be considered offline.

        """

        self.locked = False

    def openStorage(self, base_path: str) -> None:
        """Enter storage mode.

        Parameters
        ----------
        base_path: str
            A string that uniquely identifies this storage
            operation.

        """

        pass

    def closeStorage(self) -> None:
        """Leave storage mode."""

        pass

    def reset(self) -> None:
        """Reset the channel.

        This is an abstract method required to be implemented by a
        subclass.

        A resetted channel is considered completely empty with a
        similar buffer layout like after creation. It resets only its
        contents and not other properties like mode or freqeuency.

        """

        raise NotImplementedError("reset")

    def isEmpty(self) -> bool:
        """Check if the active step is empty.

        This is an abstract method required to be implemented by a
        subclass.

        """

        raise NotImplementedError("isEmpty")

    def getStepCount(self) -> int:
        """Get the number of steps in this channel's buffers.

        This is an abstract method required to be implemented by a
        subclass.

        The cound returned by this method should not include the offline
        step or any other special steps, but only those created by a
        measurement.

        """

        raise NotImplementedError("getStepCount")

    # PUBLIC USER API
    @staticmethod
    def getByName(name: str) -> AbstractChannel:
        """Get the object of a channel by its name.

        A shortcut for the get() function in this module.

        """

        return get(name)

    # Shortcuts for the watch()/unwatch() function in this module.
    watch = staticmethod(watch)
    unwatch = staticmethod(unwatch)

    def listen(self, watcher, **params) -> None:
        """Listen to callbacks of this channel.

        A shortcut for watching on this very specific channel only. It
        uses internally the watch() function to register itself, it
        therefore impose the same restrictions of unique watchers. The
        equivalent call to watch() is:

        watch(watcher, channel=self, ...)

        Parameters
        ----------
        watcher: object
            An object that is notified of callbacks on this
            channel.
        **params
            same keywords as watch()

        """
        watch(watcher, channel=self, **params)

    # Synonym for unwatch()
    unlisten = unwatch

    def subscribe(self, obj: Subscriber) -> None:
        """Subscribe to this channel.

        Add a subscriber object to this channel that receives callbacks.

        Parameters
        ----------
        obj: Subscriber
            The Subscriber object to be added

        """

        if self.name not in _channels:
            raise RuntimeError("channel is closed")

        self.subscribers.append(obj)

        obj._channel_subscriber_step_index = Step.CURRENT

        self._notify("subscriberAdded")

    def getSubscribedStep(self, obj: Subscriber) -> Step:
        """
        Return the subscribed step for an object.

        Parameters
        ----------
        obj: Subscriber
            The subscriber object the step should be returned for.

        Returns
        -------
        Step
            An integer either describing a specific step or a special
            step index.

        """
        return obj._channel_subscriber_step_index

    def setSubscribedStep(self, obj: Subscriber, step_index: Step) -> None:
        """Set the subscribed step for an object.

        Parameters
        ----------
        obj: Subscriber
            The subscriber object the step should be changed for.
        step_index: Step
            An integer containing either a specific step or
            a special index.

        """

        if obj._channel_subscriber_step_index is step_index:
            return

        obj._channel_subscriber_step_index = step_index

    def unsubscribe(self, obj: Subscriber) -> None:
        """Unsubscribe from this channel.

        Removes a subscriber object from this channel to no longer
        receive any callbacks.

        Parameters
        ----------
        obj: Subscriber
            The Subscriber object to be removed

        """

        self.subscribers.remove(obj)

        self._notify("subscriberRemoved")

    def hintDisplayArgument(self, key: str, value: Any) -> None:
        """Suggest a non-default value for arguments of display devices.

        Parameters
        ----------
        key: str
            The argument key as a string in object notation to
            hint, e.g. display.plot.steps
        value: Any
            The value this argument should use, None to unset

        """

        if value is None:
            del self.display_arguments[key]

        else:
            self.display_arguments[key] = value

    def hintDisplayArguments(self, arg_map: Mapping[str, Any]) -> None:
        """Suggest non-default values for arguments of display devices.

        This method allows to set several arguments at once compared to
        hintDisplayArgument(key, value).

        Args:
            arg_map: A mapping of key, value pairs to set in the same
                format as corresponding calls to hintDisplayArgument.

        """

        for key, value in arg_map.items():
            self.hintDisplayArgument(key, value)

    def setDirect(self) -> None:
        """Change this channel to direct mode.

        This method may not be called during a measurement.
        """

        if self.locked:
            raise RuntimeError("channel is locked")

        if self.mode == AbstractChannel.DIRECT_MODE:
            return
        elif self.mode == AbstractChannel.COMPUTING_MODE:
            self._stopComputing()
        elif self.mode == AbstractChannel.INTEGRATING_MODE:
            self._stopIntegrating()
        elif self.mode == AbstractChannel.REMOTE_MODE:
            pass

        self.mode = AbstractChannel.DIRECT_MODE

        return self

    def setComputing(
        self, kernel: Callable, input_channels: Sequence["AbstractChannel"]
    ) -> None:
        """Change this channel to computing mode.

        This method may not be called during a measurement.

        In computing mode a kernel is evaluated for samples emitted by a
        set of channels called inputs of the computing channel. The
        computing channel saves the last value generated by one of the
        input channels and executes the kernel once it has a value for
        each channel, passing these as arguments. The return value is
        added to this channel.

        If one input channel emits values at a higher frequency than
        another input channel, samples of the faster one will therefore
        be skipped.

        Args:
            kernel: The callable object acting as the kernel of this
                computing channel.
            input_channels: A sequence of channel objects that serve as
                inputs for the computing kernel.
        """

        if self.locked:
            raise RuntimeError("channel is locked")

        if self.mode == AbstractChannel.COMPUTING_MODE:
            return
        elif self.mode == AbstractChannel.INTEGRATING_MODE:
            self._stopIntegrating()

        self.mode = AbstractChannel.COMPUTING_MODE

        self.kernel = kernel
        self.input_channels = input_channels.copy()
        self.input_subscriber = []

        if len(input_channels) == 1:
            # We do a slight optimization for single argument kernels
            # here since this use case is very common and we can save
            # quite a few allocations for this path.
            self.input_stack = None

            subscr = Subscriber()
            subscr.channel = input_channels[0]
            subscr.dataAdded = self._computing_single_dataAdded

            input_channels[0].subscribe(subscr)
            self.input_subscriber.append(subscr)

        else:
            self.input_stack = [None] * len(self.input_channels)

            i = 0

            for ch in self.input_channels:
                subscr = Subscriber()
                subscr.channel = ch
                # This creates a new callable object that will call the
                # supplied function with the additional given argument.
                # Here this is used to supply a channel index to the
                # dataAdded callback, so that all channels can use the
                # same method while still being able to distinguish
                # between each.
                subscr.dataAdded = functools.partial(
                    self._computing_multiple_dataAdded, index=i
                )

                ch.subscribe(subscr)

                self.input_subscriber.append(subscr)
                i += 1

        return self

    def setIntegrating(
        self, kernel: Callable, input_channels: Sequence[AbstractChannel]
    ) -> None:
        """Change this channel to integrating mode.

        This method may not be called during a measurement.

        In integrating mode a kernel is evaluated at the end of each
        step passing all data generated by a set of channels called
        inputs as arguments. The result value is set to this channel.

        Args:
            kernel: The callable object acting as the kernel of this
                integrating channel.
            input_channels: A sequence of channel objects that serve as
                inputs for the integrating kernel.
        """

        if self.locked:
            raise RuntimeError("channel is locked")

        if self.mode == AbstractChannel.INTEGRATING_MODE:
            return
        elif self.mode == AbstractChannel.COMPUTING_MODE:
            self._stopComputing()

        self.mode = AbstractChannel.INTEGRATING_MODE

        self.kernel = kernel
        self.input_channels = input_channels.copy()

        self.setFrequency(AbstractChannel.STEP_SAMPLES)

        return self

    def setRemote(self, host: str, name: str) -> None:
        """Change this channel to remote mode."""

        self.mode = AbstractChannel.REMOTE_MODE

    def setHint(self, new_hint: Union[int, str]) -> None:
        """Change the data hint of this channel.

        Args:
            new_hint: Data hint to set to either as one of the magic
                constants or the respective string describing it

        Returns:
            The channel object itself to allow call chaining.

        Raises:
            ValueError: Unknown hint string.
        """

        if isinstance(new_hint, str):
            new_hint = AbstractChannel.getHintConstant(new_hint)

        self.hint = new_hint

        return self

    def setFrequency(self, new_freq: Union[int, str]) -> None:
        """Change the frequency of this channel.

        This method may not be called during a measurement

        Args:
            new_freq: Frequency to set to either as one of the magic
                constants or the respective string describing it

        Returns:
            The channel object itself to allow call chaining.

        Raises:
            ValueError: Unknown frequency string.
        """

        if self.locked:
            raise RuntimeError("channel is locked")

        if isinstance(new_freq, str):
            new_freq = AbstractChannel.getFrequencyConstant(new_freq)

        self.freq = new_freq

        return self

    def close(self) -> None:
        """Close this channel.

        After closing, no data should be added to a channel. It is no
        longer visible and can no longer be subscribed to.

        Channels depending on this channel will still hold a reference
        to the closed channel until closed themselves.
        """

        if self.mode == AbstractChannel.REMOTE_MODE:
            pass
        elif self.mode == AbstractChannel.COMPUTING_MODE:
            self._stopComputing()
        elif self.mode == AbstractChannel.INTEGRATING_MODE:
            self._stopIntegrating()

        self._notify("channelClosed")

        del _channels[self.name]

    def setHeaderTag(self, tag: str, value: str) -> None:
        """Set a custom header tag.

        These tags are meant to describe metadata and are saved in a
        channel-specific header in storage mode.

        Arguments:
            tag: A string containing the tag name
            value: A string containing the tag value
        """

        self.header_tags["X-" + tag] = value

    def addMarker(self, text: str) -> None:
        """Write a custom marker.

        Args;
            text: A string containing the marker to write.
        """

        pass

    def copyLayoutFrom(self, ch: "AbstractChannel") -> None:
        """Copy step layout of another channel.

        The buffer layout requrements are replicated into this channel.
        This is most useful for static channels which may come with some
        arbitrary layout, but also useful on projections in general.
        The data in this channel is not meant to be changed, but the
        buffers are guaranteed to be able to cope with the same step
        indices.

        This is an abstract method required to be implemented by a
        subclass.

        Arguments:
            ch: The channel object to copy the structure from
        """

        raise NotImplementedError("copyLayoutFrom")

    def dump(self, step: int = CURRENT_STEP, fp: IO = None) -> None:
        """Dump channel data.

        This is an abstract method required to be implemented by a
        subclass.
        """

        pass

    def getData(self, step_index: int = CURRENT_STEP) -> Any:
        """Get channel data.

        This is an abstract method required to be implemented by a
        subclass.

        Args:
            step_index: An integer describing the step to return

        Returns:
            Requested channel data.
        """

        raise NotImplementedError("getData")

    def setData(self, value: Any) -> None:
        """Set channel data.

        This is an abstract method required to be implemented by a
        subclass.

        Args:
            value: Data the current step of this channel is set to.
        """

        raise NotImplementedError("setData")

    def addData(self, value: Any) -> None:
        """Add channel data.

        This is an abstract method required to be implemented by a
        subclass.

        Args:
            value: Data to be added to the current step of this channel.
        """

        raise NotImplementedError("addData")

    def clearData(self) -> None:
        """Clear channel data.

        This is an abstract method required to be implemented by a
        subclass.
        """

        raise NotImplementedError("clearData")
