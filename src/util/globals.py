# Temporal resolution, 24 per beat
resolution = 24

# All have 4/4 time-signature
beats_per_bar = resolution * 4

# NOTE: add 1 for silence (0th index)
num_pitches = 128 + 1

note_lengths = {
    'full': resolution,
    'half': resolution//2,
    'quarter': resolution,
    'eigth': resolution//8,
    'sixteenth': resolution//16
}

import torch
# DEVICE = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
DEVICE = 'cpu'