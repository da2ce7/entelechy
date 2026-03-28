# src/shared/retrieval_future.py
"""Host-side result retrieval protocol (ADR-010)."""
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class RetrievalFuture(Protocol):
    """Protocol for host-side observation of device results (ADR-010).

    Each backend implements this protocol for its native completion
    signaling mechanism. The renderer owns padding-stripping — .result()
    returns the unpadded numpy array with logical shape.
    """

    @property
    def node_id(self) -> str:
        """The RetrievalNode's node_id that produced this future."""
        ...

    def wait(self) -> None:
        """Block until the data is host-accessible. Idempotent."""
        ...

    def result(self) -> NDArray[np.floating]:
        """Return the retrieved data as an unpadded numpy array.

        Implicitly calls .wait() if the transfer has not completed.
        The renderer applies padding-stripping before storing the result.
        """
        ...

    def release(self) -> None:
        """Signal that the host has finished consuming the result.

        The renderer may reclaim physical memory backing this result
        after release() is called.
        """
        ...
