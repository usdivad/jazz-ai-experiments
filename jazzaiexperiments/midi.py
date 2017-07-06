"""Jazz AI Experiments - MIDI functions."""

import datetime
import os
import re

# import h5py
# import keras
import mido
# import numpy as np


def construct_input_filepath(tune_name, data_dir):
    """Create a path to a MIDI file, given a tune name."""
    midi_filename = "{}.mid".format(tune_name)
    midi_filepath = os.path.join(data_dir, midi_filename)
    return midi_filepath


def construct_output_filepath(tune_name, data_dir):
    """Create a path to a MIDI output file, given a tune name."""
    curr_datetime = str(datetime.datetime.now())
    curr_datetime = re.sub("\W+", "", curr_datetime)
    out_filename = "out_{}_{}.mid".format(tune_name, curr_datetime)
    out_filepath = os.path.join(data_dir, out_filename)
    return out_filepath


def load_melody_from_file(filepath):
    """Load the melody track from a MIDI file."""
    midi_file = mido.MidiFile(filepath)
    midi_track = midi_file.tracks[0]
    return midi_track


def extract_note_messages(track):
    """Extract note messages from a MIDI track."""
    return [msg for msg in track
            if msg.type == "note_on" or msg.type == "note_off"]


def extract_note_pairs(track):
    """Extract note on/off pairs from a MIDI track."""
    notes = extract_note_messages(track)
    note_pairs = [(notes[i], notes[i + 1]) for i, _ in enumerate(notes[:-1])
                  if notes[i].type == "note_on" and
                  notes[i + 1].type == "note_off" and
                  notes[i].note == notes[i + 1].note]
    return note_pairs


def normalize_velocities(note_pairs, interval=10):
    """Normalize note velocities."""
    for note_on, note_off in note_pairs:
        note_on.velocity = note_on.velocity - (note_on.velocity % interval)
    return note_pairs


def get_note_event_keys():
    """Get keys for note events."""
    # Don't use note off velocity to shrink possibilities,
    # and don't use note off pitch because it's the same as note on pitch
    keys = ("noteon_pitch", "noteon_velocity",
            "noteon_time", "noteoff_time")
    return keys


def create_note_events(note_pairs):
    """Create note events from note pairs.

    This is the base data structure for note manipulation.
    """
    note_events = [(note_on.note, note_on.velocity,
                    note_on.time, note_off.time)
                   for note_on, note_off in note_pairs]
    return note_events


def create_note_set(note_events):
    """Create note set from note events."""
    note_set = sorted(list(set(note_events)))
    return note_set


def map_note_to_int(note_events):
    """Map notes to unique ints, and back."""
    note_set = create_note_set(note_events)
    note_to_int = dict((n, i) for i, n in enumerate(note_set))
    int_to_note = dict((i, n) for i, n in enumerate(note_set))
    return (note_to_int, int_to_note)


def split_subsequences(note_events, seq_length=10):
    """Split note events into subsequences (to feed into the model)."""
    data_input = []  # "X"
    data_output = []  # "y"
    note_to_int, int_to_note = map_note_to_int(note_events)

    for i in range(len(note_events) - seq_length):
        seqs_input = note_events[i:i + seq_length]
        seqs_output = note_events[i + seq_length]
        data_input.append([note_to_int[note] for note in seqs_input])
        data_output.append(note_to_int[seqs_output])

    return (data_input, data_output)


def write_file(note_events, output_filepath,
               midi_source_filepath=None):
    """Write note events to MIDI file.

    Convert the sequence of note tuples into a sequence of MIDI notes,
    and then write to MIDI file.
    """
    # Create MIDI file and track
    midi_file_out = mido.MidiFile()
    midi_track_out = mido.MidiTrack()
    midi_file_out.tracks.append(midi_track_out)

    # Append "headers" (track name, tempo, key, time signature)
    if midi_source_filepath is not None:
        midi_track = load_melody_from_file(midi_source_filepath)
        for message in midi_track[:4]:
            midi_track_out.append(message)
        else:
            pass

    # Add notes
    prev_time = 0
    note_events_keys = get_note_event_keys()

    # Note times get all bunched together, so we stretch them out
    # a little bit manually here...
    time_multiplier = 2  # Art Pepper - Anthropology
    time_multiplier = 0.02  # Coleman Hawkins - Body and Soul

    for note in note_events:
        # Note on/off pairs
        note = dict((note_events_keys[i], note[i]) for i, _ in enumerate(note))
        noteon_time = int(note["noteon_time"] * time_multiplier)
        noteoff_time = int(note["noteoff_time"] * time_multiplier)
        curr_time_noteon = prev_time + noteon_time
        curr_time_noteoff = prev_time + noteoff_time
        # prev_time = curr_time_noteoff
        message_noteon = mido.Message("note_on",
                                      note=note["noteon_pitch"],
                                      velocity=note["noteon_velocity"],
                                      time=curr_time_noteon)
        message_noteoff = mido.Message("note_off",
                                       note=note["noteon_pitch"],
                                       velocity=note["noteon_velocity"],
                                       time=curr_time_noteoff)
        midi_track_out.append(message_noteon)
        midi_track_out.append(message_noteoff)

    # Save file to disk
    midi_file_out.save(output_filepath)

    # for message in midi_track_out[4:20]:
    #     print(message)
    return output_filepath