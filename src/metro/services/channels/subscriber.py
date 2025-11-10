from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


class Subscriber(object):
    """Interface for channel subscribers.

    A channel subscriber listens to data changes in the respective
    channel by using callbacks. Note that like the channels API in
    general, these methods are always called on the main thread!

    """

    def dataSet(self, d: Any) -> None:
        """Callback for when channel data is set.

        Setting channel data replaces all current data for the active
        step. The subscriber should therefore discard all previous
        data received through dataAdded. There is no dataCleared call
        in this case.

        Parameters
        ----------
        d: Any
            data the current step was set to. Do not modify this
            object as it shared across all subscribers, make a
            copy in this case.

        """

        pass

    def dataAdded(self, d: Any) -> None:
        """Callback for when channel data is added.

        Adding channel data appends this to the rows already emitted in
        the active step.

        Parameters
        ----------
        d: Any
            data added to the current step. Do not modify this
            object as it shared across all subscribers, make a
            copy in this case.

        """

        pass

    def dataCleared(self) -> None:
        """Callback for when channel data is cleared.

        Clearing channel data causes the buffer for the active step to
        be empty. This is also called on each step boundary to prepare
        for the new step (even for measurements with multiple scans due
        to performance considerations!).

        """

        pass
