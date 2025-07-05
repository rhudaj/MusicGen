import numpy as np
from .types import PianoState
from .globals import num_pitches
import pypianoroll as pr


def one_hot_encode_pianoroll(pianoroll: np.ndarray) -> np.ndarray:
    '''
    piannoroll: 2D matrix (shape = [time, 128 pitches])

    RETURNS:
    A one-hot encoding (shape = [time, 129]).
    For each timestamp:
    - If notes were played: take the max value as the 1.
    - If no notes were played: make the 0th index as the 1 (represents the silence note)
    '''
    time_steps = pianoroll.shape[0]

    # Find max pitch indices for all timesteps at once
    max_indices = np.argmax(pianoroll, axis=1)

    # Check which timesteps have any notes (max value > 0)
    has_notes = np.max(pianoroll, axis=1) > 0

    # Create output array
    output = np.zeros((time_steps, num_pitches+1)) # for the silence note

    # Set silence for timesteps with no notes
    output[~has_notes, 0] = 1.0

    # Set max pitch for timesteps with notes (shift by 1 for silence padding)
    note_timesteps = np.where(has_notes)[0]
    output[note_timesteps, max_indices[note_timesteps] + 1] = 1.0

    return output


# Conversions: pianoroll <--> piano-states
def convert_pianoroll_to_piano_states(pianoroll: np.ndarray) -> list[PianoState]:
    """
    Convert pianoroll to continuous (note: int, duration: int) pairs
    """

    # STEP 1: get one-hot-encoded version
    pianoroll = one_hot_encode_pianoroll(pianoroll)

    states = []
    i = 0

    while i < len(pianoroll):
        # Find active notes
        active_notes = np.where(pianoroll[i] > 0)[0]

        if len(active_notes) == 0:
            note_value = 0  # Silence
        else:
            note_value = float(active_notes[-1])  # Take highest note

        # Count duration
        duration = 1
        while (i + duration < len(pianoroll) and
               np.array_equal(pianoroll[i], pianoroll[i + duration])):
            duration += 1

        states.append(PianoState((note_value, duration)))
        i += duration

    return states


def convert_states_to_pianoroll(
    piano_states: list[PianoState],
    num_pitches=128
):
    """
    Convert piano states (note, duration) back to pianoroll format

    Args:
        piano_states: torch.Tensor of shape (num_events, 2) where each row is [note, duration]
        num_pitches: number of pitch columns in output pianoroll

    Returns:
        np.ndarray: pianoroll of shape (total_time, num_pitches)
    """
    # Convert to numpy
    states = np.array(piano_states)

    # Calculate total time needed
    total_time = int(np.sum(states[:, 1]))

    # Create empty pianoroll
    pianoroll = np.zeros((total_time, num_pitches))

    # Fill in the pianoroll
    current_time = 0
    for note, duration in states:
        note, duration = int(note), int(duration)
        # Only fill if it's not silence (note > 0)
        if note != 0:
            pianoroll[current_time:current_time+duration, note] = 1
        # Extend time
        current_time += duration

    return pianoroll


def output_piannoroll_to_midi(
    piano_states: list[PianoState],
    instrument: str,
    name = "output"
):

    piannoroll = convert_states_to_pianoroll(piano_states)

    # Option 1: If your pianoroll has velocity values
    bass_track = pr.StandardTrack(
        pianoroll=piannoroll,  # should have values 0-127
        program=32,  # MIDI program for acoustic bass
        is_drum=False,
        name=instrument
    )

    # Otherwise, you should use pr.BinaryTrack

    # Create tempo array
    tempo_array = np.full(piannoroll.shape[0], 120.0)

    # Create Multitrack
    multitrack = pr.Multitrack(
        tracks=[bass_track],
        tempo=tempo_array,
        resolution=24,
        downbeat=None
    )

    # Export to MIDI file
    multitrack.write(f'{name}.mid')