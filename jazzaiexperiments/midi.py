"""Jazz AI Experiments - MIDI functions."""

import datetime
import os
import re

# import h5py
# import keras
import mido
# import numpy as np

from . import db


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


def extract_note_pairs(track, mode):
    """Extract note on/off pairs from a MIDI track."""
    notes = extract_note_messages(track)
    note_pairs = []

    if mode == "drums":
        # TODO: Test that this method still works for the old modes
        # `single_melody` and `single_melody_harmony`
        note_off_time = 32
        for i, note in enumerate(notes):
            if note.type == "note_on":
                # Register our note on
                note_on = note

                # Just construct a note off manually
                note_off = mido.Message("note_off",
                                        channel=0,
                                        note=note.note,
                                        velocity=note.velocity,
                                        time=note_off_time)

                # Add the pair to the mix
                note_pairs.append((note_on, note_off))

                # # Find the earliest subsequent note off for this note on,
                # # and then create a note pair out of it
                # # note_off_time = 0
                # for other_note in notes[i:]:
                #     if other_note.type == "note_off" and \
                #        other_note.note == note.note and \
                #        other_note.time != 0:
                #         note_off = other_note
                #         # note_off.time = note_off_time
                #         # note_off_time = 0
                #         note_pairs.append((note_on, note_off))
                #         break
                #     # else:
                #     #     note_off_time += other_note.time
    elif mode == "single_melody" or mode == "single_melody_harmony":
        # Old method where we don't look beyond the note immediately following
        # the note on event
        note_pairs = [(notes[i], notes[i + 1])
                      for i, _ in enumerate(notes[:-1])
                      if notes[i].type == "note_on" and
                      notes[i + 1].type == "note_off" and
                      notes[i].note == notes[i + 1].note]

    return note_pairs


def normalize_velocities(note_pairs, interval=10):
    """Normalize note velocities."""
    for note_on, note_off in note_pairs:
        note_on.velocity = note_on.velocity - (note_on.velocity % interval)
    return note_pairs


def get_note_event_keys(mode="single_melody"):
    """Get keys for note events."""
    if mode == "single_melody" or mode == "drums":
        # Don't use note off velocity to shrink possibilities,
        # and don't use note off pitch because it's the same as note on pitch
        return ("noteon_pitch", "noteon_velocity",
                "noteon_time", "noteoff_time")
    elif mode == "single_melody_harmony":
        return ("noteon_pitch", "noteon_velocity",
                "noteon_time", "noteoff_time",
                "chord")
    return ()


def create_note_events(note_pairs, mode="single_melody", chords=[]):
    """Create note events from note pairs.

    This is the base data structure for note manipulation.
    """
    note_events = []
    if mode == "single_melody" or mode == "drums":
        note_events = [(note_on.note, note_on.velocity,
                        note_on.time, note_off.time)
                       for note_on, note_off in note_pairs]
    elif mode == "single_melody_harmony":
        if len(chords) < len(note_pairs):
            print("ERROR: Number of chords must match number of melody notes!")
            return note_events

        note_events = [(note_on.note, note_on.velocity,
                        note_on.time, note_off.time,
                        chords[i])
                       for i, (note_on, note_off) in enumerate(note_pairs)]

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


def note_event_to_dict(note, mode):
    """Convert a note event tuple to a dict."""
    note_events_keys = get_note_event_keys(mode)
    return dict((note_events_keys[i], note[i]) for i, _ in enumerate(note))


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
               mode="single_melody",
               time_multiplier=1,
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
    note_events_keys = get_note_event_keys(mode=mode)

    # Note times get all bunched together, so we stretch them out
    # a little bit manually here...
    # TODO: Revisit this to make it more robust
    # time_multiplier = 2  # Art Pepper - Anthropology
    # time_multiplier = 0.02  # Coleman Hawkins - Body and Soul
    # time_multiplier = 2.5  # John Coltrane - Giant Steps
    time_multiplier = time_multiplier

    if mode == "single_melody" or mode == "single_melody_harmony":
        # Harmony (chord) settings
        prev_chord = "NC"
        chord_velocity = 64

        for note in note_events:
            # Construct messages for note on/off pairs
            note = dict((note_events_keys[i], note[i])
                        for i, _ in enumerate(note))
            noteon_time = int(note["noteon_time"] * time_multiplier)
            noteoff_time = int(note["noteoff_time"] * time_multiplier)
            curr_time_noteon = prev_time + noteon_time
            curr_time_noteoff = prev_time + noteoff_time
            # prev_time = curr_time_noteoff
            message_noteon = mido.Message("note_on",
                                          channel=0,
                                          note=note["noteon_pitch"],
                                          velocity=note["noteon_velocity"],
                                          time=curr_time_noteon)
            message_noteoff = mido.Message("note_off",
                                           channel=0,
                                           note=note["noteon_pitch"],
                                           velocity=note["noteon_velocity"],
                                           time=curr_time_noteoff)

            # Append note on event
            midi_track_out.append(message_noteon)

            # Write harmony (chords) as well
            if mode == "single_melody_harmony":
                curr_chord = note["chord"]
                if curr_chord != prev_chord:
                    curr_pitches = db.chord_to_midi_pitches(curr_chord)
                    prev_pitches = db.chord_to_midi_pitches(prev_chord)

                    # Add note ons for current chord
                    for pitch in curr_pitches:
                        message = mido.Message("note_on",
                                               channel=1,
                                               note=pitch,
                                               velocity=chord_velocity,
                                               time=0)  # time=curr_noteon
                        midi_track_out.append(message)

                    # Add note offs for previous chord
                    for pitch in prev_pitches:
                        message = mido.Message("note_off",
                                               channel=1,
                                               note=pitch,
                                               velocity=chord_velocity,
                                               time=0)  # time=curr_noteon
                        midi_track_out.append(message)

                prev_chord = curr_chord

            # Append the note off event (we do this after appending harmony
            # so that the harmony lines up with the current note on)
            midi_track_out.append(message_noteoff)
    elif mode == "drums":
        # note_idx = 0
        # while note_idx < len(note_events):
        #     note_ons = []
        #     note_offs = []
        #     note_event = note_events[note_idx]
        #     note_ons.append(note_event[0])
        #     note_ons.append(note_event[1])
        #     note_idx += 1
        #     note_event = note_events[note_idx]
        #     while note
        #     # last_simultaneous_note_idx = note_idx
        #     # for i in range

        noteoff_messages = []
        # prev_noteoff = []
        for i, note in enumerate(note_events[:-1]):
            # Construct current note on and off
            note = dict((note_events_keys[i], note[i])
                        for i, _ in enumerate(note))
            noteon_time = int(note["noteon_time"] * time_multiplier)
            noteoff_time = int(note["noteoff_time"] * time_multiplier)
            curr_time_noteon = prev_time + noteon_time
            curr_time_noteoff = prev_time + noteoff_time
            message_noteon = mido.Message("note_on",
                                          channel=0,
                                          note=note["noteon_pitch"],
                                          velocity=note["noteon_velocity"],
                                          time=0)
            message_noteoff = mido.Message("note_off",
                                           channel=0,
                                           note=note["noteon_pitch"],
                                           velocity=note["noteon_velocity"],
                                           time=0)

            # Add note on to track before note offs
            # midi_track_out.append(message_noteon)

            # If necessary, add note offs from any previous notes
            # (which may or may not have had simultaneous onsets)
            if noteon_time > 0:
                noteoff_messages[0].time = curr_time_noteon
                midi_track_out.extend(noteoff_messages)
                noteoff_messages = []

            # Add note on to track AFTER note offs
            midi_track_out.append(message_noteon)

            # Add note off to buffer
            noteoff_messages.append(message_noteoff)

            # Only add note offs to track if next note is not simultaneous
            # next_note = note_events[i + 1]
            # next_note = dict((note_events_keys[i], next_note[i])
            #                  for i, _ in enumerate(next_note))
            # if next_note["noteon_time"] != 0:
            #     for i, message in enumerate(noteoff_messages):
            #         if i > 0:
            #             message.time = 0  # Make note offs simultaneous too
            #         midi_track_out.append(message)
            #         noteoff_messages = []

    # Save file to disk
    midi_file_out.save(output_filepath)

    # for message in midi_track_out[4:20]:
    #     print(message)
    return (midi_file_out, output_filepath)


def calculate_note_times_seconds(input_filepath):
    """Calculate times of note onsets (in seconds)."""
    midi_file = mido.MidiFile(input_filepath)
    midi_track = load_melody_from_file(input_filepath)
    tempo = midi_track[1].tempo
    # ppq = midi_track[3].clocks_per_click
    # n32 = midi_track[3].notated_32nd_notes_per_beat
    ppq = midi_file.ticks_per_beat
    note_times = [mido.tick2second(msg.time, ppq, tempo)
                  for msg in midi_track if "note" in msg.type]
    note_times_summed = []
    for i, t in enumerate(note_times):
        note_times_summed.append(sum(note_times[:i]) + t)
    return note_times_summed
