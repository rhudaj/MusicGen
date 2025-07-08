import os, glob
import pypianoroll as pr
from typing import Optional
from util.globals import num_pitches
from util.convert import convert_pianoroll_to_piano_states
from util.types import Song, PianoState, Instrument

data_dir = 'data/lpd_17_cleansed'

def get_all_npz_files():
	"""Get all .npz files from the nested directory structure"""
	pattern = os.path.join(data_dir, "*", "*", "*", "*", "*.npz")
	return glob.glob(pattern)

def get_songs() -> list[Song]:
	songs: list[Song] = []

	# Get file data
	filepaths = get_all_npz_files()
	print(f'Found {len(filepaths)} total files')

	# Get info on each file
	import json
	with open('data/midi_info.json') as f:
		midi_info = json.load(f)

	for path in filepaths:
		id = path.split('/')[-1].split('.')[0]
		songs.append(
			Song(path, id, midi_info[id])
		)

	return songs


def load_multi_track(file_path) -> pr.Multitrack:
	return pr.load(f'{data_dir}/{file_path}')

def get_track_by_instrument(
	multi_track: pr.Multitrack,
	instrument='Bass'
) -> Optional[pr.Track]:

	found_tracks = list(filter(
		lambda t: t.name == instrument,
		multi_track.tracks
	))

	if not found_tracks or len(found_tracks) == 0:
		return None

	return found_tracks[0]

def get_samples(
	songs: list[Song],
	desired_instrument: Instrument,
	max_samples = 100,
) -> list[list[PianoState]]:
	'''
	Collect piano-roll data for specific instruments
	'''

	samples: list[list[PianoState]] = []

	for song in songs:

		if len(samples) >= max_samples:
			break

		try:
			multi_track = pr.load(song.fullpath)
		except:
			print(f'Error loading file: {song.fullpath}')
			continue

		desired_track = get_track_by_instrument(multi_track, desired_instrument)
		if desired_track:
			# we found it
			desired_pianoroll = desired_track.pianoroll
			samples.append(
				convert_pianoroll_to_piano_states(desired_pianoroll)
			)

	return samples