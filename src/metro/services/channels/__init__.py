"""This module implements channels, which are undirectional connections
between a provider and an arbitrary number of subscribers. The
provider broadcasts data to all subscribers by setting (replacing the
current content) or adding the data content of a channel.

There are both complete implementations available (currently only
NumericChannel) and it is possible to inherit from AbstractChannel and
implement various abstract methods.

In most cases related to data acquisition, NumericChannel is the
optimal choice.

"""

from __future__ import annotations

from enum import Enum

from .abstract import AbstractChannel

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterable


class Mode(Enum):
    """Channel mode constants.

    The mode of a channel specifies how samples are generated.

    Attributes
    ----------
    DIRECT: Mode
        Samples are generated directly by manual calls to the
        public API.
    COMPUTING: Mode
        The channel computes samples by using samples emitted
        by one or more other channel(s), once per sample emitted.
    INTEGRATING: Mode
        Similar to the COMPUTING mode, but limited to once per
        step and scan. A channel operating in this mode will
        always use STEP as frequency property.
    REMOTE: Mode
        A channel from another Metro instance is replicated over
        a socket connection.

    """

    DIRECT = 1
    COMPUTING = 2
    INTEGRATING = 3
    REMOTE = 4


class Hint(Enum):
    """Data hint constants.

    This optional property of a channel can be used to suggest a
    suitable presentation for this channel's data to the user.

    Attributes
    ----------
    UNKNOWN: Hint
        No hint is given, the user may have to decide on its
        own on how to display the data.
    ARBITRARY: Hint
        The channel does not necessarily contain numeric
        data and therefore it should not be attempted to
        be displayed with the generic devices.
    INDICATOR: Hint
        Only the most recent sample should be displayed at
        a time. This hint is useful when complete data sets
        are added to a channel instead of point-by-point.
    WAVEFORM: Hint
        The probed variable varies over time and should be
        presented on a value by value base.
    HISTOGRAM: Hint
        The distribution of a certain variable is measured
        at possibly irregular intervals. Not the individual
        values are of interest, but a histogram of said
        distribution.

    """

    UNKNOWN = 1
    ARBITRARY = 2
    INDICATOR = 3
    WAVEFORM = 4
    HISTOGRAM = 5


class Frequency(Enum):
    """Channel frequency constants.

    The frequency of a channel specifies how samples are emitted to
    subscribers and what data layout they can expect from this channel.
    It can also change the semantics of certain method calls.

    Attributes
    ----------
    CONTINUOUS: Frequency
        Samples are generated at arbitrary intervals and are
        grouped in steps. For several scans, the samples are
        appended to the same step.
    STEP: Frequency
        There is always exactly ONE sample at the end of each
        step over all scans. This mode is used for statistics
        over one step like number of counts an average and also
        internally once a channel is switched to integrating
        mode. If used with a direct channel, the sample may be
        generated at the stopped signal.
    SCHEDULED: Frequency
        This mode completely ignores step boundaries and simply
        emits samples at a known interval.

    """

    CONTINUOUS = 1
    STEP = 2
    SCHEDULED = 3


class Step(Enum):
    """Magic step indices

    These magic constants are for addressing relative step indices like
    the current or all at once. Channel implementation may use the
    assumption that any index below 0 is a special index.

    Attributes
    ----------
    CURRENT: Step
        ?
    ALL: Step
        ?
    """

    CURRENT = -1
    ALL = -2


class ChannelManager(object):
    watchers = {}
    channels = {}

    def get(self, name: str) -> AbstractChannel:
        """Get a channel by name.

        Parameters
        ----------
        name: str
            A string containing the complete channel name.

        Returns
        -------
        AbstractChannel
            The requested channel object.

        Raises
        ------
        KeyError
            No channel found with that name.

        """

        return self.channels[name]

    def getAll(self) -> Iterable[AbstractChannel]:
        """Get all current channels.

        Returns
        -------
        Iterable
            A dict view containing all channel objects.

        """

        return self.channels.values()

    def query(
        self,
        hint: Hint | None = None,
        freq: Frequency | None = None,
        type_: type | None = None,
        shape: int | None = None,
    ) -> list[AbstractChannel]:
        """Query channels with certain parameters.

        The query can be limited to a certain hint, freq or type.

        Parameters
        ----------
        hint: Hint or None
            Data hint to query for to either as one of the magic
            constants or the respective string describing it, None as
            wildcard.
        freq: Frequency or None
            Frequency to query for to either as one of the magic
            constants or the respective string describing it, None as
            wildcard.
        type_: type or None
            Limits all results to a certain type of channel, None as
            wildcard.
        shape: int or None
            Shape paramter to query for or None as wildcard. This
            property is actually not part of AbstractChannel, but used
            in the common implementation NumericChannel. Using this
            argument on non-compatible channels will simply exclude
            them from the results.

        Returns
        -------
        list
            A list of strings with the channel names complying with the
            specified query parameters.

        """

        if isinstance(hint, str):
            hint = Hint[hint]

        elif isinstance(hint, int):
            hint = Hint(hint)

        if isinstance(freq, str):
            freq = Frequency[freq]

        elif isinstance(freq, int):
            freq = Frequency(freq)

        res = []

        for name, channel in self.channels.items():
            hit = True

            if hint is not None and hint is not channel.hint:
                hit = False

            if freq is not None and freq is not channel.freq:
                hit = False

            if type_ is not None and not isinstance(channel, type_):
                hit = False

            if shape is not None:
                try:
                    if shape != channel.shape:
                        hit = False
                except AttributeError:
                    hit = False

            if hit:
                res.append(name)

        return res

    def watch(
        self,
        watcher: object,
        hint: Hint | None = None,
        freq: Frequency | None = None,
        type_: type | None = None,
        shape: int | None = None,
        channel: AbstractChannel | None = None,
        callbacks: list[str] | None = None,
    ) -> None:
        """Watch the channel list for certain parameters.

        A watcher is notified whenever a channel is opened or closed. A
        watcher object may register once for a set of parameters. Any
        subsequent call will only change the parameters.

        Parameters
        ----------
        watcher: object
            An object that is notified of changes to the channel
            list conforming to the parameters. The exact callbacks are
            implementation-specific for a channel, but there are generic
            ones supported by AbstractChannel.
        hint: Hint or None
            Data hint to exclusively watch for to either as one of
            the magic constants or the respective string describing it,
            None as wildcard.
        freq: Frequency or None
            Frequency to exclusively watch for to either as one of
            the magic constants or the respective string describing it,
            None as wildcard.
        shape: int or None
            Shape paramter to watch for or None as wildcard. The
            same restrictions apply here as for the shape argument of
            query().
        type_: type or None
            An optional type object specifying a channel type to
            exclusively watch for or None as wildcard.
        channel: AbstractChannel or None
            An optional object specifying a specific channel to
            exclusively watch for or None as wildcard.
        callbacks: list or None
            An optional list of callbacks to watch for
            exclusively.

        """

        if isinstance(hint, str):
            hint = Hint[hint]

        elif isinstance(hint, int):
            hint = Hint(hint)

        if isinstance(freq, str):
            freq = Frequency[freq]

        elif isinstance(freq, int):
            freq = Frequency(freq)

        self.watchers[watcher] = (hint, freq, type_, shape, channel, callbacks)

    def unwatch(self, watcher: object) -> None:
        """Stop watching the channel list.

        Parameters
        ----------
        watcher: object
            The object to stop watching the channel list.

        Raises
        ------
        KeyError
            Invalid watcher object.

        """

        del self.watchers[watcher]


def sortByDependency(
    all_channels: Iterable[AbstractChannel],
) -> list[AbstractChannel]:
    """Sort a channel list by its dependencies.

    Some channels may depend on other channels, such as a channel in
    computing or integrating mode, depend on its arguments. This
    function takes a list of channels and sorts it in such a way that
    all dependencies are satisfied using a topological sort algorithm.
    This is for example used by the measuring controller while calling
    all channels at the beginning/end of each step/scan.

    Parameters
    ----------
    all_channels: Iterable
        list of channel objects

    Returns
    -------
    list
        A list containing the same channels as all_channels, but sorted
        by dependency.

    Raises
    ------
    RuntimeError
        Circular channel dependency detected

    """

    # First we build the graph for our topological sort. This is a list
    # containing a tuple with each channel object and a set containing
    # the channel objects this channel depends on.
    graph = []

    # For this we loop over every channel-channel combination and check
    # for dependency. We could omit this step if channels would deliver
    # a list with all dependencies themselves, but this is "only" O(n²)
    # keeps the interface much more clean.
    for outer_channel in all_channels:
        deps = set()

        for inner_channel in all_channels:
            if outer_channel is inner_channel:
                continue

            if outer_channel.dependsOn(inner_channel):
                deps.add(inner_channel)

        graph.append((outer_channel, deps))

    # This list contains our channels sorted by dependency.
    final_channels = []

    # This set is used in the topological sort and contains all
    # dependencies that are already satisfied.
    provided_deps = set()

    # We loop until there are no unsorted channels left
    while graph:
        # This becomes our new graph at the end of this loop iteration
        # with all channels that are not yet taken care of
        remaining_graph = []

        # This flag ensures that we emit a channel on each iteration. If
        # not we have a cyclic dependency and fail!
        emitted = False

        # Loop through all remaining channels
        for channel, deps in graph:
            # Check of all dependencies of this channel has already
            # been taken care of
            if deps.issubset(provided_deps):
                # If yes, add it to our final list and to the satisfied
                # dependencies (emit it)
                final_channels.append(channel)
                provided_deps.add(channel)
                emitted = True
            else:
                # If not, add it to our remaining graph
                remaining_graph.append((channel, deps))

        # If this flag is still false, we did not emit a single channel
        # and therefore had unsatisfiable dependencies (cycles).
        if not emitted:
            raise RuntimeError("circular channel dependency detected")

        # Switch our graph to all remaining channels.
        # This operation is equivalent of removing channels from our
        # original graph, but it is not possible to change a list being
        # iterated upon.
        graph = remaining_graph

    return final_channels
