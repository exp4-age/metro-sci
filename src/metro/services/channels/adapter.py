from __future__ import annotations

from .abstract import AbstractChannel

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


class ChannelAdapter(AbstractChannel):
    """Adapter for the channel interface.

    This class implements stubs for all abstract methods of a channel.
    It is intended for custom channel implementations that only want to
    plug into specifics of the channel framework without actually
    providing the complete feature set.

    As an example, the sources/dld_rd device uses such a channel to
    obtain the current storage location and stream the raw TDC opcode
    data there.

    """

    def reset(self) -> None:
        pass

    def isEmpty(self) -> bool:
        return True

    def getStepCount(self) -> int:
        return 0

    def getData(self, step_index: int) -> None:
        return None

    def setData(self, value: Any) -> None:
        pass

    def addData(self, value: Any) -> None:
        pass

    def clearData(self) -> None:
        pass
