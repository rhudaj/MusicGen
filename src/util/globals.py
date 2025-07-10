# Temporal resolution – pulses per beat (1 beat = 1 quarter note)
resolution = 24

# All have 4/4 time-signature
beats_per_bar = resolution * 4

# NOTE: add 1 for silence (0th index)
num_pitches = 128 + 1

note_lengths = {
    'full': resolution*4,
    'half': resolution*2,
    'quarter': resolution,
    'eigth': resolution//2,
    'sixteenth': resolution//4
}

import torch
# DEVICE = 'cpu'
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else "cpu"