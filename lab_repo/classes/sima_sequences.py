from sima.sequence import _WrapperSequence, _MotionCorrectedSequence
import numpy as np
import inspect
import sima
import sima.sequence
import cv2

from scipy.ndimage.filters import gaussian_filter
from numba import float32, int32, int16, njit


class _SpatialFilterSequence(_WrapperSequence):
    """Sequence for gaussian blurring and clipping each frame.

    Parameters
    ----------
    base : Sequence

    """

    def __init__(self, base):
        super(_SpatialFilterSequence, self).__init__(base)

    def _transform(self, frame):
        n_channels = frame.shape[-1]
        ch_frames = []
        for i in xrange(n_channels):
            ch_frame = frame[..., i]
            filtered_frame = gaussian_filter(ch_frame, sigma=3)
            clipped = np.clip(filtered_frame,
                              np.nanpercentile(filtered_frame, 50),
                              np.nanpercentile(filtered_frame, 99.5))

            ch_frames.append(clipped)

        return np.stack(ch_frames, axis=-1)

    def _get_frame(self, t):
        frame = self._base._get_frame(t)
        return np.array(map(self._transform, frame))

    def __iter__(self):
        for frame in self._base:
            yield self._transform(frame)

    @property
    def shape(self):
        return self._base.shape

    def __len__(self):
        return len(self._base)

    def _todict(self, savedir=None):
        return {
            '__class__': self.__class__,
            'base': self._base._todict(savedir),
        }


class _SplicedSequence(_WrapperSequence):
    """Sequence for splicing together nonconsecutive frames
    Parameters
    ----------
    base : Sequence
    times : list of frame indices of the base sequence
    """

    def __init__(self, base, times):
        super(_SplicedSequence, self).__init__(base)
        self._base_len = len(base)
        self._times = times

    def __iter__(self):
        try:
            for t in self._times:
                # Not sure if np.copy is needed here (see _IndexedSequence)
                yield np.copy(self._base._get_frame(t))
        except NotImplementedError:
            if self._indices[0].step < 0:
                raise NotImplementedError(
                    'Iterating backwards not supported by the base class')
            idx = 0
            for t, frame in enumerate(self._base):
                try:
                    whether_yield = t == self._times[idx]
                except IndexError:
                    raise StopIteration
                if whether_yield:
                    # Not sure if np.copy is needed here (see _IndexedSequence)
                    yield np.copy(frame)
                    idx += 1

    def _get_frame(self, t):
        return self._base._get_frame(self._times[t])

    def __len__(self):
        return len(self._times)

    def _todict(self, savedir=None):
        return {
            '__class__': self.__class__,
            'base': self._base._todict(savedir),
            'times': self._times
        }

        
sima.sequence.__dict__.update(
    {k: v for k, v in locals().items() if
        inspect.isclass(v) and issubclass(v, sima.sequence._WrapperSequence)})
