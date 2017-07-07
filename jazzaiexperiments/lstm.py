"""Jazz AI Experiments - LSTM functions."""

import datetime
import os
import re

# import h5py
import keras
import numpy as np

from . import db
from . import midi


def create(x_shape, y_shape,
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


def setup_callbacks(tune_name, data_dir):
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


def fit_model(model, x, y,
              num_epochs=200,
              batch_size=32,
              callbacks=[]):
    """Fit the LSTM model."""
    model.fit(x, y,
              epochs=num_epochs,
              batch_size=batch_size,
              callbacks=callbacks)
    return model


def load_model_weights(model, weights_filepath):
    """Load weights for a model."""
    model.load_weights(weights_filepath)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


def train_on_midi_input(tune_name,
                        midi_data_dir="../data/midi/",
                        checkpoints_data_dir="../data/models/",
                        input_filepath=None,
                        weights_filepath=None,
                        db_beats_filepath=None,
                        db_melody_filepath=None,
                        seq_length=10,
                        num_epochs=100,
                        mode="single_melody"):
    """Build and train an LSTM from an input MIDI file.

    To load weights, just pass in a weights_filepath.
    If no additional training is needed, just set num_epochs to 0.
    """
    # Load MIDI file
    if input_filepath is None:
        input_filepath = midi.construct_input_filepath(tune_name,
                                                       midi_data_dir)
    midi_track = midi.load_melody_from_file(input_filepath)
    note_pairs = midi.extract_note_pairs(midi_track, mode)
    note_pairs = midi.normalize_velocities(note_pairs, interval=10)

    # Get harmony data if need be
    chords = []
    if mode == "single_melody_harmony":
        chords = db.get_harmony_for_melody(db_beats_filepath,
                                           db_melody_filepath)

    # Create note events
    note_events = []
    note_events = midi.create_note_events(note_pairs,
                                          mode=mode,
                                          chords=chords)
    print("Created {} note events from "
          "{} using mode {}".format(len(note_events), input_filepath, mode))

    # Format note data to feed into network
    note_set = midi.create_note_set(note_events)
    seqs_input, seqs_output = midi.split_subsequences(note_events,
                                                      seq_length=seq_length)
    num_seqs = len(seqs_input)
    seq_length = len(seqs_input[0])
    num_unique_notes = len(note_set)
    x, y = format_midi_for_lstm(seqs_input, seqs_output,
                                num_seqs=num_seqs,
                                seq_length=seq_length,
                                num_unique_notes=num_unique_notes)
    print("Formatted note data ({} seqs of length {}, "
          "{} unique notes)".format(num_seqs, seq_length, num_unique_notes))

    # Create LSTM
    model = create(x.shape, y.shape, num_units=256, dropout_rate=0.2)
    print("Created model")

    # Train LSTM, or load from weights
    if weights_filepath is not None:
        model = load_model_weights(model, weights_filepath)
        print("Loaded weights from {}".format(weights_filepath))
    else:
        print("No weights loaded (to load weights, "
              "specify a `weights_filepath`)")

    if num_epochs > 0:
        callbacks = setup_callbacks(tune_name, checkpoints_data_dir)
        model = fit_model(model, x, y,
                          num_epochs=num_epochs,
                          batch_size=32,
                          callbacks=callbacks)
        print("Trained model over {} epochs".format(num_epochs))
    else:
        print("No training needed (`num_epochs` specified as 0)")

    return (model, note_events, input_filepath)


def construct_input_seq(note_events, seq_length,
                        random_seed=False):
    """Construct an input sequence, given note events and sequence length."""
    seqs_input, seqs_output = midi.split_subsequences(note_events,
                                                      seq_length=seq_length)
    seq_in = seqs_input[0]
    if random_seed:
        seq_in = seqs_input[np.random.randint(len(seqs_input))]
    return seq_in


def format_midi_for_lstm(data_input, data_output,
                         num_seqs, seq_length, num_unique_notes):
    """Reshape + normalize input, and convert output to one-hot encoding."""
    # Reshape input sequences into form [samples, time steps, features]
    x = np.reshape(data_input, (num_seqs, seq_length, 1))

    # Normalize to 0-1 range
    x = x / float(num_unique_notes)

    # Convert output to one-hot encoding
    y = keras.utils.np_utils.to_categorical(data_output)

    return (x, y)


def generate_notes(model, note_events, seq_in,
                   num_notes_to_generate=100,
                   batch_size=32,
                   add_seed_to_output=False):
    """Generate notes given a model, note events, and input sequence."""
    notes_out = []
    _, int_to_note = midi.map_note_to_int(note_events)
    num_unique_notes = len(midi.create_note_set(note_events))

    if add_seed_to_output:
        seq_in_notes = [int_to_note[i] for i in seq_in]
        notes_out.extend(seq_in_notes)

    for i in range(num_notes_to_generate):
        # Reshape and normalize
        x = np.reshape(seq_in, (1, len(seq_in), 1))  # Reshape
        x = x / float(num_unique_notes)  # Normalize

        # Make the prediction
        pred = model.predict(x, batch_size=batch_size, verbose=0)

        # Get output note
        note_idx = np.argmax(pred)
        note = int_to_note[note_idx]

        # Add output note to list
        notes_out.append(note)

        # Add output note to input sequence, and move forward by one note
        seq_in.append(note_idx)
        seq_in = seq_in[1:len(seq_in)]

    return notes_out


def generate_midi_output(model, note_events,
                         mode="single_melody",
                         num_notes_to_generate=100,
                         batch_size=32,
                         time_multiplier=1,
                         random_seed=False,
                         add_seed_to_output=False,
                         output_filepath=None,
                         midi_source_filepath=None,
                         tune_name="output",
                         data_dir="../data/output/"):
    """Generate note output, given a trained model."""
    # Construct input sequence
    seq_in = construct_input_seq(note_events, model.input_shape[1],
                                 random_seed=random_seed)
    print("Constructed input sequence: {}".format(seq_in))

    # Generate the notes!
    num_notes = num_notes_to_generate
    notes_out = generate_notes(model, note_events, seq_in,
                               num_notes_to_generate=num_notes,
                               batch_size=batch_size,
                               add_seed_to_output=add_seed_to_output)
    print("Generated {} notes".format(num_notes))

    # Write output to MIDI file
    if output_filepath is None:
        output_filepath = midi.construct_output_filepath(tune_name, data_dir)
    midi.write_file(notes_out, output_filepath,
                    mode=mode,
                    time_multiplier=time_multiplier,
                    midi_source_filepath=midi_source_filepath)
    print("Wrote to MIDI file at {}".format(output_filepath))
    return (notes_out, output_filepath)
