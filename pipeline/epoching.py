import numpy as np
import mne


def epoch_raw(
    raw: mne.io.BaseRaw,
    target_sfreq: float = 200.0,
    window_sec: float = 10.0,
) -> np.ndarray:
    """Resample and slice continuous EEG into fixed-length non-overlapping epochs.

    Args:
        raw: MNE Raw object (any sampling rate).
        target_sfreq: Target sampling rate after resampling.
        window_sec: Epoch window length in seconds.

    Returns:
        np.ndarray of shape (M, C, T) where
            M = number of full windows,
            C = number of channels,
            T = int(target_sfreq * window_sec).
    """
    if raw.info["sfreq"] != target_sfreq:
        raw = raw.copy().resample(target_sfreq, verbose=False)

    data = raw.get_data()  # (C, total_samples)
    n_channels, total_samples = data.shape
    samples_per_window = int(target_sfreq * window_sec)

    n_epochs = total_samples // samples_per_window
    if n_epochs == 0:
        raise ValueError(
            f"Recording too short ({total_samples / target_sfreq:.1f}s) "
            f"for {window_sec}s windows."
        )

    # Trim to exact multiple of window size and reshape
    trimmed = data[:, : n_epochs * samples_per_window]
    epochs = trimmed.reshape(n_channels, n_epochs, samples_per_window)
    epochs = epochs.transpose(1, 0, 2)  # (M, C, T)

    return epochs
