"""Jazz AI Experiments - database functions."""

import os

import music21
import pandas as pd


def construct_filepath(tune_name,
                       data_type,  # "beats", "melody"
                       data_dir="../data/db/"):
    """Construct a filepath to a CSV containing a subset of the database."""
    filename = "{}_{}.csv".format(tune_name, data_type)
    filepath = os.path.join(data_dir, filename)
    return filepath


def read_file(filepath):
    """Read a database CSV."""
    data = pd.read_csv(filepath)
    return data


def get_harmony_for_melody(beats_filepath, melody_filepath):
    """Get the underlying chord for each melody note."""
    data_beats = read_file(beats_filepath).dropna(subset=["chord"])
    data_melody = read_file(melody_filepath)
    chords = []

    for i, melevt in data_melody.iterrows():
        # Get beats that came before current melody event (i.e. note)
        beats = data_beats[data_beats.onset <= melevt.onset]
        if len(beats) < 1:
            chords.append("NC")
            continue

        # Get most recent chord
        most_recent_beat = beats.iloc[-1:]
        # print("{}: {}".format(i, len(beats)))
        # print(most_recent_beat.chord)
        chord = most_recent_beat.chord
        chord = chord[chord.keys()[0]] if len(chord.keys()) > 0 else "NC"
        chords.append(chord)

    return chords


def chord_to_music21_chord(chord):
    """Convert a chord from the Weimar db to be music21-compatible."""
    new_chord = chord

    # Use "m" instead of "-" to indicate minor chords
    new_chord = new_chord.replace("-", "m")

    # Use "maj" instead of "j" to indicate maj7 chords
    new_chord = new_chord.replace("j", "maj")

    # Use "sus4add7" instead of "sus7" for suspended 7th chords
    # TODO: Get it to add a dominant 7th ("sus4add7" adds a major 7th)
    #       (probably could use music21.harmony.addNewChordSymbol())
    new_chord = new_chord.replace("sus7", "sus4")

    # According to music21 docs:
    # "if a root or bass is flat, the ‘-‘ must be used, and NOT ‘b’.
    # However, alterations and chord abbreviations are specified normally
    # with the ‘b’ and ‘#’ signs."
    if len(new_chord) > 1 and new_chord[1] == "b":
        new_chord_arr = list(new_chord)
        new_chord_arr[1] = "-"
        new_chord = "".join(new_chord_arr)

    # Convert sharps and flats to go before interval; e.g. "B-79b" to "B-7b9"
    if len(new_chord) > 0 and (new_chord[-1] == "b" or new_chord[-1] == "#"):
        new_chord_arr = list(new_chord)
        if len(new_chord_arr) > 2:
            accidental = new_chord_arr[-1]
            new_chord_arr[-1] = new_chord_arr[-2]
            new_chord_arr[-2] = accidental
        new_chord = "".join(new_chord_arr)

    return new_chord


def chord_to_midi_pitches(chord):  # TODO: Add optional octave argument
    """Convert a chord symbol to MIDI pitches."""
    if chord == "NC":
        return []
    chord_name = chord_to_music21_chord(chord)
    chord_symbol = music21.harmony.ChordSymbol(chord_name)
    return [pitch.midi for pitch in chord_symbol.pitches]
