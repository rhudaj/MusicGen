import torch
from torch.utils.data import Dataset, DataLoader

from ..util.types import Instrument, Song, PianoState
from .load import get_songs, get_samples

class InstrumentDataset(torch.utils.data.Dataset):
    '''
    Only pianoroll's for a specific instrument
    NOTE: move batches to device during training (to avoid overloading the GPU)
    '''
    songs: list[Song]
    sequences: list[list[PianoState]]

    # Accessed during training
    seq_tensors: list[torch.Tensor]
    target_tensors: list[torch.Tensor]

    def __init__(self,
        instrument: Instrument,
        max_samples: int = 100
    ):

        # Load data from file
        self.songs = get_songs()
        self.sequences = get_samples(self.songs, instrument, max_samples)

        print(f'Got {len(self.sequences)} total sequences for instrument "{instrument}"')

        # Convert to tensors
        seq_tensors = [
            torch.Tensor(seq)
            for seq in self.sequences
        ]

        # Create input/target pairs
        self.seq_tensors = [ seq[:-1].float() for seq in seq_tensors ]
        self.target_tensors = [ seq[1:] for seq in seq_tensors ]

    def __len__(self):
        return len(self.seq_tensors)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        ''' Returns: (sequence tensor, target sequence tensor)'''
        return self.seq_tensors[idx], self.target_tensors[idx]