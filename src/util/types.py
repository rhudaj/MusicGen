import numpy as np
from dataclasses import dataclass

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
