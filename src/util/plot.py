import numpy as np

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

import pypianoroll as pr

from .types import PianoState, NoteSample
from .convert import convert_states_to_pianoroll
from .globals import beats_per_bar, resolution

def plot_pianoroll(
    pianoroll: np.ndarray,
    ax = None,
    title="",
    tick_resolution=resolution
):
	'''Plot the piano sequence'''
	if not ax:
		_, ax = plt.subplots(1, 1, figsize=(15,8))

	img = ax.imshow(pianoroll.T,
                 aspect='auto',
                 origin='lower',
                 cmap='binary',
                 interpolation='none')

	ax.set_title(title)
	ax.set_xlabel(f'Time (tick-resolution = {resolution/tick_resolution} beats)')
	ax.set_ylabel('Pitch')
	plt.colorbar(img, ax=ax)

	# set the x-axis tick locator
	ax.xaxis.set_major_locator(MultipleLocator(tick_resolution))

	# Convert tick labels from time-steps to beats
	ticks = ax.get_xticks()
	ax.set_xticklabels([int(tick / resolution) for tick in ticks])

	ax.grid(visible=True, axis='x')

	return ax

def plot_piano_states(
	piano_states: list[PianoState],
	ax=None,
	title=""
):
	"""Convert piano states to pianoroll and plot using existing function"""
	pianoroll = convert_states_to_pianoroll(piano_states)
	return plot_pianoroll(pianoroll, ax, title)


def plot_note_sample_probs(
	note_samples: list[NoteSample]
):
	chosen_notes = np.array([ ns.note for ns in note_samples ])
	note_probs = np.array([ ns.probs for ns in note_samples ])

	# Create figure if no axis provided
	fig, ax = plt.subplots(1, 1, figsize=(12, 8))

	# Plot probability heatmap
	im = ax.imshow(note_probs.T, aspect='auto', origin='lower', cmap='Blues')

	# Highlight chosen notes with red dots
	time_steps = np.arange(len(chosen_notes))
	ax.scatter(time_steps, chosen_notes, color='red', s=50, alpha=0.8, label='Chosen Notes')

	# Add colorbar
	plt.colorbar(im, ax=ax, label='Probability')

	# Labels and formatting
	ax.set_xlabel('Time Step')
	ax.set_ylabel('Note (MIDI)')
	ax.set_title('Note Sample Probabilities')
	ax.legend()

	ax.set_ybound(
		lower=chosen_notes.min() - 1,
		upper=chosen_notes.max() + 1
	)

	# Optional: Add grid for better readability
	ax.grid(True, alpha=0.3)

	return ax


def plot_track(data: pr.Track, desired_instrument: str, with_bars=True, line_spacing_bars=1, start_beat=0):
	'''
	Wrapper for pr.plot_track that adds vertical lines for bars
 	'''

	num_bars = int(data.pianoroll.shape[0] / beats_per_bar)

	# plot the track (with bars)
	fig, ax = plt.subplots(figsize=(12, 6))
	pr.plot_track(data, ax=ax)

	# Add vertical lines for bar boundaries
	if with_bars:
		for bar in range(1, num_bars+1, line_spacing_bars):
			bar_position = bar * beats_per_bar
			ax.axvline(x=bar_position, color='red', linestyle='--', alpha=0.7, linewidth=1)

	# Add labels
	ax.set_title(f'{desired_instrument} Track - {num_bars} bars')
	ax.set_xlabel('Time (beats)')
	plt.tight_layout()
	plt.show()
