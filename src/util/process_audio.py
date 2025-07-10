import numpy as np

def quantize_pianoroll(piano_roll: np.ndarray,
                       resolution: int,
                       strength: float = 1.0,
                       preserve_note_length: bool = True) -> np.ndarray:
    """
    Quantize a piano roll to a specified resolution grid.

    Parameters:
    -----------
    piano_roll : np.ndarray
        2D array of shape (time_steps, num_pitches) where non-zero values
        represent active notes
    resolution : int
        Quantization resolution in time steps (e.g., 4 for quarter notes if
        your time resolution is 16th notes)
    strength : float, optional
        Quantization strength from 0.0 (no quantization) to 1.0 (full quantization)
        Default is 1.0
    preserve_note_length : bool, optional
        If True, maintains original note durations. If False, notes may be
        shortened/lengthened. Default is True.

    Returns:
    --------
    np.ndarray
        Quantized piano roll with same shape as input
    """

    time_steps, num_pitches = piano_roll.shape
    quantized_roll = np.zeros_like(piano_roll)

    # Process each pitch separately
    for pitch in range(num_pitches):
        pitch_data = piano_roll[:, pitch]

        # Find note onsets (where note starts from silence)
        onsets = []

        i = 0
        while i < len(pitch_data):
            if pitch_data[i] > 0:
                # Found start of a note
                onset_time = i
                velocity = pitch_data[i]

                # Find the end of this note
                note_end = i
                while note_end < len(pitch_data) and pitch_data[note_end] > 0:
                    note_end += 1

                note_length = note_end - onset_time
                onsets.append((onset_time, velocity, note_length))
                i = note_end
            else:
                i += 1

        # Quantize each onset
        for onset_time, velocity, note_length in onsets:
            # Find nearest quantization grid point
            grid_point = round(onset_time / resolution) * resolution

            # Apply quantization strength
            if strength < 1.0:
                quantized_onset = int(onset_time + strength * (grid_point - onset_time))
            else:
                quantized_onset = int(grid_point)

            # Ensure we don't go out of bounds
            quantized_onset = max(0, min(quantized_onset, time_steps - 1))

            # Determine note end time
            if preserve_note_length:
                # Keep original note length
                quantized_end = min(quantized_onset + note_length, time_steps)
            else:
                # Keep original end time (note length may change)
                original_end = onset_time + note_length
                quantized_end = min(original_end, time_steps)

            # Place the quantized note
            for t in range(quantized_onset, quantized_end):
                if t < time_steps:
                    quantized_roll[t, pitch] = velocity

    return quantized_roll
