import torch
from torch.utils.data import Dataset, DataLoader

from util.types import Instrument, Song, PianoState
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

    def __init__(self,
        instrument: Instrument,
        max_samples: int = 100,
        quantize=True,
        randomize=True,
    ):

        # Load data from file
        self.songs = get_songs()
        self.sequences = get_samples(
            self.songs,
            instrument,
            max_samples,
            quantize,
            randomize=randomize
        )

        print(f'Got {len(self.sequences)} total sequences for instrument "{instrument}"')

        # Convert to tensors
        self.seq_tensors = [
            torch.Tensor(seq).float()
            for seq in self.sequences
        ]

    def __len__(self):
        return len(self.seq_tensors)

    def __getitem__(self, idx) -> torch.Tensor:
        ''' Returns: sequence tensor'''
        return self.seq_tensors[idx]

def collate_sequences(batch: list[torch.Tensor]):
    """
    Custom collate function for Batch Creation

    Takes individual sequence tensors and packs them efficiently for RNN processing.

    Args:
        batch: list of sequence tensors

    Returns:
        tuple: (packed_inputs, batch_size)
    """
    # Sort by length (makes pack_sequence more efficient)
    sorted_sequences = sorted(batch, key=len, reverse=True)

    # Pack sequences
    packed_inputs = torch.nn.utils.rnn.pack_sequence(sorted_sequences, enforce_sorted=True)

    batch_size = len(batch)

    return packed_inputs, batch_size

def get_dataloader(
    dataset: InstrumentDataset,
    batch_size: int,
):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_sequences,
        pin_memory=False,
    )