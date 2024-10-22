import os
import pretty_midi
import numpy as np

def analyze_midi(file_path):
    try:
        midi_data = pretty_midi.PrettyMIDI(file_path)
        file_length = midi_data.get_end_time()
        
        note_lengths = []
        total_notes = 0
        for instrument in midi_data.instruments:
            if not instrument.is_drum:  # Exclude drum tracks
                total_notes += len(instrument.notes)
                for note in instrument.notes:
                    note_lengths.append(note.end - note.start)
        
        return file_length, note_lengths, total_notes
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return 0, [], 0

def calculate_averages(directory):
    total_file_length = 0
    all_note_lengths = []
    total_notes = 0
    file_count = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.midi') or file.endswith('.mid'):
                file_path = os.path.join(root, file)
                file_length, note_lengths, notes_count = analyze_midi(file_path)
                if file_length > 0:
                    total_file_length += file_length
                    all_note_lengths.extend(note_lengths)
                    total_notes += notes_count
                    file_count += 1
    
    if file_count > 0:
        avg_file_length = total_file_length / file_count
        avg_note_length = np.mean(all_note_lengths) if all_note_lengths else 0
        avg_notes_per_file = total_notes / file_count
        return avg_file_length, avg_note_length, avg_notes_per_file
    else:
        return 0, 0, 0

# Replace with the path to your extracted MAESTRO v1 MIDI files
maestro_directory = '/home/junha/dna_classification/Music/data/maestro-v1.0.0/2017'
avg_file_length, avg_note_length, avg_notes_per_file = calculate_averages(maestro_directory)

print(f"Average length of MIDI files: {avg_file_length:.2f} seconds")
print(f"Average length of notes: {avg_note_length:.4f} seconds")
print(f"Average number of notes per file: {avg_notes_per_file:.2f}")