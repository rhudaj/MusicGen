# External Imports
import torch as torch

from util.globals import beats_per_bar, num_pitches, DEVICE
from util.plot import plot_note_sample_probs
from util.convert import output_piannoroll_to_midi
from models import MusicLSTM
from models.train import train_batched
from models.infer import sample_notes
from dataset.dataset import InstrumentDataset, get_dataloader

# MODEL
model = MusicLSTM(
    hidden_size=256,
    num_pitches=num_pitches+1,  # 0-128 notes (including silence at 0)
    num_layers=2,
    bidirectional=True,
    dropout=0.1
).to(DEVICE)

print(f"# Parameters: {sum(p.numel() for p in model.parameters())}")

# DATASET
dataset = InstrumentDataset(
	instrument='Bass',
	max_samples=20,
)
trainloader = get_dataloader(dataset, 5)


# TRAINING
train_batched(
    model,
    trainloader,
    num_epochs=200,
    lr=0.001
)


# INFERENCE
seq = dataset[0]
predictions = sample_notes(
    model,
    start_event=torch.Tensor(seq[0]).to(DEVICE),
    length=beats_per_bar*1,
    temperature=0.3
)

plot_note_sample_probs(predictions.note_samples)
output_piannoroll_to_midi(
    predictions.piano_states,
    instrument='Guitar',
    name='generated_guitar2'
)
