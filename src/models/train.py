# MODEL TRAINING
from util.globals import DEVICE
from .rnn.model import MusicRNN
from .lstm.model import MusicLSTM
import torch

'''
NOTE train_step vs. `train_step_vectorized`

They're mathematically identical but the vectorized version is more efficient.
=> Vectorized is faster due to parallel computation instead of sequential loops

Both do the same thing:

- Forward pass: Both process the entire sequence through the RNN with proper hidden state updates
- Loss computation: Both sum losses across all time steps
- Gradients: Identical gradient values and backpropagation
- Model parameters: Produce exactly the same trained model

How they are different:

- train_step: Python loop processing one time step at a time
- train_step_vectorized_correct: Single vectorized operation processing all time steps at once
'''

def train_step(
	model: MusicRNN,
	sequence: torch.Tensor, # (seq_len, num_features)
	note_criterion: torch.nn.CrossEntropyLoss,
	duration_criterion: torch.nn.MSELoss,
	optimizer: torch.optim.Adam,
):
	"""
	A single step of training - trains the model on 1 sequence
	Train with CrossEntropyLoss for note classification and MSELoss for duration regression
	"""
	optimizer.zero_grad()
	hidden = model.init_hidden()
	total_loss = 0.0

	for i in range(len(sequence) - 1):
		# Current event: [note, duration]
		input_event = sequence[i:i+1].float()  			# Shape: (1, 2)
		target_note = sequence[i+1, 0].long()      		# Note as integer for classification
		target_duration = sequence[i+1, 1:2].float()  	# Duration as float for regression

		# Forward pass
		note_logits, duration_pred, hidden = model(input_event, hidden)

		# Classification loss for notes
		note_loss = note_criterion(note_logits.squeeze(0), target_note)

  		# Regression loss for durations
		duration_loss = duration_criterion(duration_pred.squeeze(0), target_duration)

		# Combine losses
		total_loss += note_loss + duration_loss

	total_loss.backward()
	optimizer.step()
	return total_loss.item()

def train_step_vectorized(
    model: MusicRNN,
    sequence: torch.Tensor,
    note_criterion: torch.nn.CrossEntropyLoss,
    duration_criterion: torch.nn.MSELoss,
    optimizer: torch.optim.Adam,
):
    optimizer.zero_grad()

    # Prepare inputs and targets
    inputs = sequence[:-1].float()  # Shape: (seq_len-1, 2)
    targets = sequence[1:]          # Shape: (seq_len-1, 2)

    # Use your existing init_hidden (without batch dimension)
    hidden = model.init_hidden()  # Shape: (num_layers, hidden_size)

    # Forward pass
    note_logits, duration_pred, _ = model(inputs, hidden)

    # Extract targets
    target_notes = targets[:, 0].long()
    target_durations = targets[:, 1:2].float()

    # Compute losses
    note_loss = note_criterion(note_logits, target_notes)
    duration_loss = duration_criterion(duration_pred, target_durations)

    total_loss = note_loss + duration_loss
    total_loss.backward()
    optimizer.step()

    return total_loss.item()

def train(
	model: MusicRNN,
	sequences: list[torch.Tensor],
	num_epochs=1000,
	lr=0.001,
) -> list[float]:
	'''
	Train over epochs on multiple sequences.

	Args:
		model:
		sequences:
		num_epochs: number of training epochs
		lr: learning rate
	Returns:
 		list[float]: loss over all epochs
 	'''


 	# Set to training mode
	model.train()

	# Training Parameters
	note_criterion = torch.nn.CrossEntropyLoss(reduction='sum')
	duration_criterion = torch.nn.MSELoss(reduction='sum')
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

	N = len(sequences)

	for epoch in range(num_epochs):
		epoch_loss = 0
		for i in range(N):
			loss = train_step_vectorized(
       			model,
          		sequences[i],
            	note_criterion,
             	duration_criterion,
              	optimizer
            )
			epoch_loss += loss
		if epoch % (num_epochs//100) == 0:
			print(f'Epoch {epoch}/{num_epochs}, Loss = ', epoch_loss)

# ------------------------------------------------------------
# FOR BATCH TRAINING
# ------------------------------------------------------------

def train_step_fully_packed(
    model: MusicLSTM,
    packed_input_batch: torch.nn.utils.rnn.PackedSequence,
    batch_size: int,
    note_criterion: torch.nn.CrossEntropyLoss,
    duration_criterion: torch.nn.MSELoss,
    optimizer: torch.optim.Optimizer,
    clip_grads = True,
    max_norm: float = 1.0,
):
    """Simplified version that works with pre-packed batches from dataloader"""
    optimizer.zero_grad()

    # Forward pass
    packed_note_logits, packed_duration_pred = model(packed_input_batch)

    # Create targets from inputs (shift by one timestep)
    # For packed sequences, we need to shift the data portion
    input_data = packed_input_batch.data[1:]  # Remove first timestep
    target_notes = input_data[:, 0].long()
    target_durations = input_data[:, 1:2].float()

    # Predictions exclude the last timestep (can't predict beyond sequence end)
    pred_note_logits = packed_note_logits.data[:-1]
    pred_durations = packed_duration_pred.data[:-1]

    # Compute totalloss
    note_loss = note_criterion(pred_note_logits, target_notes)
    duration_loss = duration_criterion(pred_durations, target_durations)
    total_loss = note_loss + duration_loss

    total_loss.backward()

    if clip_grads:
        # WHY? RNN/LSTM prone to exploding gradients, Allows higher learning rates, More stable training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

    optimizer.step()

    return total_loss.item()

def train_batched(
    model: MusicLSTM,
    dataloader: torch.utils.data.DataLoader,
    num_epochs: int = 1000,
    lr: float = 0.001,
):
    model.train()

    note_criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    duration_criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"Training on data set with n = {len(dataloader)}")

    epoch_losses = []

    for epoch in range(num_epochs):

        epoch_loss = 0

        # Serve batches of training data:
        for i, (packed_batch, batch_size) in enumerate(dataloader, 0):

            packed_batch = packed_batch.to(DEVICE)

            loss = train_step_fully_packed(
                model,
                packed_batch,
                batch_size,
                note_criterion,
                duration_criterion,
                optimizer
            )
            epoch_loss += loss

        epoch_losses.append(epoch_loss)

        # if epoch % (num_epochs // 100) == 0:
        print(f'Epoch {epoch}/{num_epochs}, Loss = {epoch_loss:.4f}')

        # EMPTY CACHE BETWEEN EPOCHS (PERFORMANCE BOOST)!
        torch.mps.empty_cache()

    return epoch_losses