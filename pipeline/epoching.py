import numpy as np
import mne


def epoch_raw(
    raw: mne.io.BaseRaw,
    target_sfreq: float = 200.0,
    window_sec: float = 10.0,
    stride_sec: float | None = None,
) -> np.ndarray:
    """Resample and slice continuous EEG into fixed-length epochs.

    Args:
        raw: MNE Raw object (any sampling rate).
        target_sfreq: Target sampling rate after resampling.
        window_sec: Epoch window length in seconds.
        stride_sec: Stride between windows in seconds. Defaults to window_sec
                     (non-overlapping). Use a smaller value for overlapping
                     epochs (e.g. stride_sec=5.0 with window_sec=10.0).

    Returns:
        np.ndarray of shape (M, C, T) where
            M = number of full windows,
            C = number of channels,
            T = int(target_sfreq * window_sec).
    """
    if stride_sec is None:
        stride_sec = window_sec

    if raw.info["sfreq"] != target_sfreq:
        raw = raw.copy().resample(target_sfreq, verbose=False)

    data = raw.get_data()  # (C, total_samples)
    n_channels, total_samples = data.shape
    samples_per_window = int(target_sfreq * window_sec)
    stride_samples = int(target_sfreq * stride_sec)

    starts = list(range(0, total_samples - samples_per_window + 1, stride_samples))
    if len(starts) == 0:
        raise ValueError(
            f"Recording too short ({total_samples / target_sfreq:.1f}s) "
            f"for {window_sec}s windows."
        )

    epochs = np.stack([data[:, s : s + samples_per_window] for s in starts])
    # epochs shape: (M, C, T)

    return epochs
