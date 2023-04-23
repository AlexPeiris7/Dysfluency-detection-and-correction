import os
from scipy.io import wavfile
import matplotlib.pyplot as plot

noise_removed_clips = '/Users/alexiopeiris/Desktop/University/FourthYear/FYP/Imp/Stuttering_detection_and_correction/stuttering_detection/preprocessing/dataset/clips'
spectrogram_dir = '/Users/alexiopeiris/Desktop/University/FourthYear/FYP/Imp/Stuttering_detection_and_correction/stuttering_detection/preprocessing/spectrograms'

# looping through the folder and sub folders containing the files to remove noise
# os.walk is a function used to loop through all the sub folders in the main folder
for root, dirs, files in os.walk(noise_removed_clips):
    for filename in files:
        # path to the file
        file_path = os.path.join(root, filename)
        if os.path.isfile(file_path) and file_path.endswith(".wav"):
            # getting the relative path of the file
            relative_path = os.path.relpath(file_path, noise_removed_clips)
            # splitting path from extension
            split_path = os.path.splitext(relative_path)[0]
            # changing extension of path file from wav to png
            relative_path = split_path
            # constructing the path to output the file
            output_spec_path = os.path.join(spectrogram_dir, relative_path)
            # Creating directories if they do not already exist
            os.makedirs(os.path.dirname(spectrogram_dir), exist_ok=True)
            print(output_spec_path)
            # Check if spectrogram already exists
            if os.path.exists(output_spec_path):
               continue
            # Missing clip
            if not os.path.exists(file_path):
                print("Missing", file_path)
                continue
            # Verify spectrogram directory exists
            os.makedirs(spectrogram_dir, exist_ok=True)
            # Read file
            sample_rate, samples = wavfile.read(file_path)
            # Generate spectrogram
            spectrogram, frequency, times, image_handle = plot.specgram(samples, Fs=sample_rate)
            # plot.specgram(samples, Fs=sample_rate)
            # Get the shape of the spectrogram(used later in the LTSM layer)
            # print(spectrogram.shape) = (129, 374)
            # splitting path from extension
            filename = os.path.splitext(filename)[0]
            # Save spectrogram to file
            plot.savefig('spectrograms/' + filename + '_spec.png')
            plot.close()

