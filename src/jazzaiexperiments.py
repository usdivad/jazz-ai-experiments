"""Jazz AI Experiments."""

import datetime
import os
import re

# import h5py
import keras
import mido
import numpy as np


def midi_create_filepath(tune_name, data_dir):
    """Create a path to a MIDI file, given a tune name."""
    midi_filename = "{}.mid".format(tune_name)
    midi_filepath = os.path.join(data_dir, midi_filename)
    return midi_filepath


def midi_load_melody_track(filepath):
    """Load the melody track from a MIDI file."""
    midi_file = mido.MidiFile(filepath)
    midi_track = midi_file.tracks[0]
    return midi_track


def midi_extract_notes(track):
    """Extract notes from a MIDI track."""
    return [msg for msg in track
            if msg.type == "note_on" or msg.type == "note_off"]


def midi_extract_note_pairs(track):
    """Extract note on/off pairs from a MIDI track."""
    notes = midi_extract_notes(track)
    note_pairs = [(notes[i], notes[i + 1]) for i, _ in enumerate(notes[:-1])
                  if notes[i].type == "note_on" and
                  notes[i + 1].type == "note_off" and
                  notes[i].note == notes[i + 1].note]
    return note_pairs


def midi_normalize_velocities(note_pairs, interval=10):
    """Normalize note velocities."""
    for note_on, note_off in note_pairs:
        note_on.velocity = note_on.velocity - (note_on.velocity % interval)
    return note_pairs


def midi_get_note_event_keys():
    """Get keys for note events."""
    # Don't use note off velocity to shrink possibilities,
    # and don't use note off pitch because it's the same as note on pitch
    keys = ("noteon_pitch", "noteon_velocity",
            "noteon_time", "noteoff_time")
    return keys


def midi_create_note_events(note_pairs):
    """Create note events from note pairs.

    This is the base data structure for note manipulation.
    """
    note_events = [(note_on.note, note_on.velocity,
                    note_on.time, note_off.time)
                   for note_on, note_off in note_pairs]
    return note_events


def midi_create_note_set(note_events):
    """Create note set from note events."""
    note_set = sorted(list(set(note_events)))
    return note_set


def midi_map_note_to_int(note_events):
    """Map notes to unique ints, and back."""
    note_set = midi_create_note_set(note_events)
    note_to_int = dict((n, i) for i, n in enumerate(note_set))
    int_to_note = dict((i, n) for i, n in enumerate(note_set))
    return (note_to_int, int_to_note)


def midi_split_subsequences(note_events, seq_length=10):
    """Split note events into subsequences (to feed into the model)."""
    data_input = []  # "X"
    data_output = []  # "y"
    note_to_int, int_to_note = midi_map_note_to_int(note_events)

    for i in range(len(note_events) - seq_length):
        seq_input = note_events[i:i + seq_length]
        seq_output = note_events[i + seq_length]
        data_input.append([note_to_int[note] for note in seq_input])
        data_output.append(note_to_int[seq_output])

    return (data_input, data_output)


def midi_format_for_lstm(data_input, data_output,
                         num_seqs, seq_length, num_unique_notes):
    """Reshape + normalize input, and convert output to one-hot encoding."""
    # Reshape input sequences into form [samples, time steps, features]
    x = np.reshape(data_input, (num_seqs, seq_length, 1))

    # Normalize to 0-1 range
    x = x / float(num_unique_notes)

    # Convert output to one-hot encoding
    y = keras.utils.np_utils.to_categorical(data_output)

    return (x, y)


def lstm_create(x_shape, y_shape,
                num_units=256,
                dropout_rate=0.2):
    """Create the basic LSTM network."""
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(num_units,
                                input_shape=(x_shape[1],
                                             x_shape[2]),
                                return_sequences=True))
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.LSTM(num_units))
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(y_shape[1], activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


def lstm_setup_callbacks(tune_name, data_dir):
    """Setup callback functions for model."""
    callbacks = []

    # Checkpoints; save weights to disk when improvement observed
    curr_datetime = str(datetime.datetime.now())
    curr_datetime = re.sub("\W+", "", curr_datetime)
    checkpoint_filename = "weights_" + str(tune_name) + "_" \
                          + str(curr_datetime) + "_{epoch:02d}_{loss:.4f}.hdf5"
    checkpoint_filepath = os.path.join(data_dir, checkpoint_filename)
    checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_filepath,
                                                 monitor="loss",
                                                 verbose=1,
                                                 save_best_only=True,
                                                 mode="min")
    callbacks.append(checkpoint)

    return callbacks


def lstm_fit_model(model, x, y,
                   num_epochs=200,
                   batch_size=32,
                   callbacks=[]):
    """Fit the LSTM model."""
    model.fit(x, y,
              epochs=num_epochs,
              batch_size=batch_size,
              callbacks=callbacks)
    return model


def lstm_train_on_input(tune_name,
                        midi_data_dir,
                        checkpoints_data_dir,
                        num_epochs=100,
                        mode="single_melody"):
    """Build and train an LSTM from an input MIDI file."""
    # Create note events
    input_filepath = midi_create_filepath(tune_name, midi_data_dir)
    midi_track = midi_load_melody_track(input_filepath)
    note_pairs = midi_extract_note_pairs(midi_track)
    note_pairs = midi_normalize_velocities(note_pairs, interval=10)
    note_events = midi_create_note_events(note_pairs)

    # Format note data to feed into network
    note_set = midi_create_note_set(note_events)
    data_input, data_output = midi_split_subsequences(note_events,
                                                      seq_length=10)
    x, y = midi_format_for_lstm(data_input, data_output,
                                num_seqs=len(data_input),
                                seq_length=len(data_input[0]),
                                num_unique_notes=len(note_set))

    # Create and train LSTM
    model = lstm_create(x.shape, y.shape, num_units=256, dropout_rate=0.2)
    callbacks = lstm_setup_callbacks(tune_name, checkpoints_data_dir)
    model = lstm_fit_model(model, x, y,
                           num_epochs=num_epochs,
                           batch_size=32,
                           callbacks=callbacks)
    return model
