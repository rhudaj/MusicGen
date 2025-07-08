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