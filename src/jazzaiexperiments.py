"""Jazz AI Experiments."""

import datetime
import os
import re

# import h5py
import keras
import mido
import numpy as np


# -------- MIDI functions --------

def midi_create_input_filepath(tune_name, data_dir):
    """Create a path to a MIDI file, given a tune name."""
    midi_filename = "{}.mid".format(tune_name)
    midi_filepath = os.path.join(data_dir, midi_filename)
    return midi_filepath


def midi_create_output_filepath(tune_name, data_dir):
    """Create a path to a MIDI output file, given a tune name."""
    curr_datetime = str(datetime.datetime.now())
    curr_datetime = re.sub("\W+", "", curr_datetime)
    out_filename = "out_{}_{}.mid".format(tune_name, curr_datetime)
    out_filepath = os.path.join(data_dir, out_filename)
    return out_filepath


def midi_load_melody_from_file(filepath):
    """Load the melody track from a MIDI file."""
    midi_file = mido.MidiFile(filepath)
    midi_track = midi_file.tracks[0]
    return midi_track


def midi_extract_note_messages(track):
    """Extract note messages from a MIDI track."""
    return [msg for msg in track
            if msg.type == "note_on" or msg.type == "note_off"]


def midi_extract_note_pairs(track):
    """Extract note on/off pairs from a MIDI track."""
    notes = midi_extract_note_messages(track)
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
        seqs_input = note_events[i:i + seq_length]
        seqs_output = note_events[i + seq_length]
        data_input.append([note_to_int[note] for note in seqs_input])
        data_output.append(note_to_int[seqs_output])

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


def midi_write_file(note_events, output_filepath,
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
        midi_track = midi_load_melody_from_file(midi_source_filepath)
        for message in midi_track[:4]:
            midi_track_out.append(message)
        else:
            pass

    # Add notes
    prev_time = 0
    note_events_keys = midi_get_note_event_keys()

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


# -------- LSTM functions --------

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


def lstm_load_weights(model, weights_filepath):
    """Load weights for a model."""
    model.load_weights(weights_filepath)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


def lstm_train_on_midi_input(tune_name,
                             midi_data_dir,
                             checkpoints_data_dir,
                             weights_filepath=None,
                             seq_length=10,
                             num_epochs=100,
                             mode="single_melody"):
    """Build and train an LSTM from an input MIDI file."""
    # Create note events
    input_filepath = midi_create_input_filepath(tune_name, midi_data_dir)
    midi_track = midi_load_melody_from_file(input_filepath)
    note_pairs = midi_extract_note_pairs(midi_track)
    note_pairs = midi_normalize_velocities(note_pairs, interval=10)
    note_events = midi_create_note_events(note_pairs)
    print("Created note events from {}".format(input_filepath))

    # Format note data to feed into network
    note_set = midi_create_note_set(note_events)
    seqs_input, seqs_output = midi_split_subsequences(note_events,
                                                      seq_length=seq_length)
    num_seqs = len(seqs_input)
    seq_length = len(seqs_input[0])
    num_unique_notes = len(note_set)
    x, y = midi_format_for_lstm(seqs_input, seqs_output,
                                num_seqs=num_seqs,
                                seq_length=seq_length,
                                num_unique_notes=num_unique_notes)
    print("Formatted note data ({} seqs of length {}, "
          "{} unique notes)".format(num_seqs, seq_length, num_unique_notes))

    # Create LSTM
    model = lstm_create(x.shape, y.shape, num_units=256, dropout_rate=0.2)
    print("Created model")

    # Train LSTM, or load from weights
    if weights_filepath is None:
        callbacks = lstm_setup_callbacks(tune_name, checkpoints_data_dir)
        model = lstm_fit_model(model, x, y,
                               num_epochs=num_epochs,
                               batch_size=32,
                               callbacks=callbacks)
        print("Trained model over {} epochs".format(num_epochs))
    else:
        model = lstm_load_weights(model, weights_filepath)
        print("Loaded weights from {}".format(weights_filepath))

    return (model, note_events, input_filepath)


def lstm_construct_input_seq(note_events, seq_length,
                             random_seed=False):
    """Construct an input sequence, given note events and sequence length."""
    seqs_input, seqs_output = midi_split_subsequences(note_events,
                                                      seq_length=seq_length)
    seq_in = seqs_input[0]
    if random_seed:
        seq_in = seqs_input[np.random.randint(len(seqs_input))]
    return seq_in


def lstm_generate_notes(model, note_events, seq_in,
                        num_notes_to_generate=100,
                        batch_size=32,
                        add_seed_to_output=False):
    """Generate notes given a model, note events, and input sequence."""
    notes_out = []
    _, int_to_note = midi_map_note_to_int(note_events)
    num_unique_notes = len(midi_create_note_set(note_events))

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


def lstm_generate_midi_output(model, note_events,
                              mode="single_melody",
                              num_notes_to_generate=100,
                              batch_size=32,
                              random_seed=False,
                              add_seed_to_output=False,
                              midi_source_filepath=None,
                              tune_name="output",
                              data_dir="../data/output/"):
    """Generate note output, given a trained model."""
    # Construct input sequence
    seq_in = lstm_construct_input_seq(note_events, model.input_shape[1],
                                      random_seed=random_seed)
    print("Constructed input sequence: {}".format(seq_in))

    # Generate the notes!
    num_notes = num_notes_to_generate
    notes_out = lstm_generate_notes(model, note_events, seq_in,
                                    num_notes_to_generate=num_notes,
                                    batch_size=batch_size,
                                    add_seed_to_output=add_seed_to_output)
    print("Generated {} notes".format(num_notes))

    # Write output to MIDI file
    output_filepath = midi_create_output_filepath(tune_name, data_dir)
    midi_write_file(notes_out, output_filepath,
                    midi_source_filepath=midi_source_filepath)
    print("Wrote to MIDI file at {}".format(output_filepath))
    return notes_out
