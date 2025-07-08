# RNN Model – Notes

## Design Choices

1. Separate Heads

- Decoupled Learning: Notes and durations have different characteristics - separate heads allow specialized feature learning for each
- Independent Optimization: Each head can learn optimal representations without interference from the other task
- Scalable Architecture: Easy to add more musical features (velocity, timbre) as additional heads

2. Output Formats

- Notes → Classification: Musical notes are discrete values (0-127), classification naturally handles this without rounding artifacts
- Duration → Regression: Durations can vary continuously and benefit from smooth interpolation between values
- Data-Appropriate: Matches the underlying nature of each musical parameter

3. Loss Functions

- CrossEntropyLoss (Notes): Optimal for multi-class discrete prediction, provides probability distributions for sampling
- MSELoss (Duration): Standard for continuous regression, penalizes large duration errors more heavily
- Combined Loss: Allows joint optimization while respecting each output type's characteristics

## About torch.nn.RNN

parameters:
	input_size (H_in) - number of expected features in the input x
 	hidden_size - number of features in the hidden state h
	num_layers - Number of recurrent layers

inputs:
	input - Tensor of shape (L, H_in) for unbatches input
	hx - Tensor of shape (D * num_layers * H_out)
		(Defaults to zeros if not provided)

Summary:
When using unbatched inputs with torch.nn.RNN:
- Input shape: (seq_len, input_size)
- Hidden shape: (num_layers, hidden_size) ← No batch dimension
When using batched inputs:
- Input shape: (seq_len, batch_size, input_size)
- Hidden shape: (num_layers, batch_size, hidden_size)


## Batch-Training with the `PyTorch` RNN

Batch processing is much more memory and compute efficient because of GPU parallelization across the batch dimension.

### Difference

**Non-Batched:**

- Process 1 sequence at a time -> 1 loss value
- Sum losses across all seq's in training loop (epoch).

**Batched**

- Process multiple sequences simultaneously (in parallel)
- Loss function receives predictions and targets for ALL sequences at once
- PyTorch automatically handles the aggregation based on the `reduction` parameter

With `reduction='sum'`, both approaches give identical results.


### Problem: Variable-Length Sequences

Can use `torch.nn.utils.rnn.pack_sequence` on batches which handles this for you.

- RNN processes the packed sequence efficiently


## About `PackedSequence`

A `PackedSequence` is PyTorch's efficient way to handle variable-length sequences in RNNs.

The problem: RNNs expect fixed-size tensors, but sequences have different lengths. Padding with zeros wastes computation.

The solution: PackedSequence contains:

- `data`: All sequence elements concatenated into one tensor
- `batch_sizes`: How many sequences are active at each timestep

The RNN processes `packed.data` sequentially, using `batch_sizes` to know when sequences end.

**Example:**

```python
# 3 sequences: [1,2,3], [4,5], [6]
sequences = [
    torch.tensor([1, 2, 3]),
    torch.tensor([4, 5]),
    torch.tensor([6])
]

packed = pack_sequence(sequences, enforce_sorted=True)
# packed.data = [1, 4, 6, 2, 5, 3]  # concatenated
# packed.batch_sizes = [3, 2, 1]    # 3 active, then 2, then 1
```

## Gradient Clipping

Prevents exploding gradients by capping their magnitude.

```python
# In your train_step_fully_packed function, after backward() but before step():
total_loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Add this line
optimizer.step()
```

**max-norm**

`max_norm=X`
	- Scales down gradients if their total magnitude exceeds `X`
		- Exceeds => All gradients get multiplied by `1/X`
		- Under => Gradients pass through unchanged
	- Keeps gradient direction, just limits size

**Larger max_norm:** Allows bigger gradient steps, so the model can learn faster.

- `max_norm=0.1` → tiny steps → slow learning
- `max_norm=5.0` → bigger steps → faster learning


**Why it helps:**

- RNNs/LSTMs prone to exploding gradients
- Allows higher learning rates
- More stable training

## Bidirectional

Creates 2 sets of states, one for each of the forward & reverse directions.

The hidden/cell-state size will now grow:

```
(self.num_layers, batch_size, self.hidden_size)
-->
(2 * self.num_layers, batch_size, self.hidden_size)
```

The outputs of the RNN/LSTM change:

```
output, output_states = self.rnn(packed_input, input_states)
```

- `output`: will contain a concat of the forward & reverse hidden states at each time step in the sequence.
- `states[0]` (h_n) will contain a concat of the final forward and reverse hidden states, respectively.
- `states[1]` (c_n) will contain a concat of the final forward and reverse hidden states, respectively.


## `train_step` vs. `train_step_vectorized`

Here was the code for `train_step` (was replaced by `train_step_vectorized`):

```python
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
```


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