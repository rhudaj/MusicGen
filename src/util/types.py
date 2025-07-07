from typing import Literal
import numpy as np
from dataclasses import dataclass

Instrument = Literal[
    'Drums',
    'Piano',
    'Chromatic Percussion',
    'Organ, Guitar',
    'Bass',
    'Strings',
    'Ensemble',
    'Brass',
    'Reed',
    'Pipe',
    'Synth Lead',
    'Synth Pad',
    'Synth Effects',
    'Ethnic',
    'Percussive',
    'Sound Effects',
]

PianoState = tuple[int, int]

@dataclass
class Song:
    # metadata
    fullpath: str
    id: str
    info: dict

@dataclass
class NoteSample:
	note: int
	probs: np.ndarray

@dataclass
class PianoStateSamples:
	piano_states: list[PianoState]
	note_samples: list[NoteSample]
