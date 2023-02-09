import noisereduce as nr
import os
from pydub import AudioSegment

# os is used to loop through the files in the folder where the wav files are
# pydub is used to load the files
# noiseruduce is the mlibrary used to reduce the noise in the wav file
# the  noise reduced file is later saved in another folder

# path containing the audio files to remove the noise
clipped_wav_path = '/Users/alexiopeiris/Desktop/University/FourthYear/FYP/Imp/Stuttering_detection_and_correction/ml-stuttering-events-dataset-main/CLIP_DIR'

# path which will contain the files which are noise removed
noise_reduced_wav_path = 'nr_clipped_files'

print('Clipped Wav Folder path found: ', os.path.isdir(clipped_wav_path))

print('Noise Reduction Process Started')

# looping through the folder and subfolders containing the files to remove noise
# os.walk is a function used to loop through all the subfolders in the main folder
for root, dirs, files in os.walk(clipped_wav_path):
    for filename in files:
        # path to the file
        file_path = os.path.join(root, filename)
        if os.path.isfile(file_path) and file_path.endswith(".wav"):
            # loading the audio file
            wav_file = AudioSegment.from_file(file_path, format="wav")
            # converting the wav file in wav format to a numpy array
            samples = wav_file.get_array_of_samples()
            # noise reduction using the imported module
            reduce_noise = nr.reduce_noise(samples, sr=wav_file.frame_rate)
            # creating new file to save the noise reduced audio file
            noise_reduced_wav = AudioSegment(
                reduce_noise.tobytes(),
                frame_rate=wav_file.frame_rate,
                sample_width=wav_file.sample_width,
                channels=wav_file.channels
            )
            # getting the relative path of the file
            relative_path = os.path.relpath(file_path, clipped_wav_path)
            # constructing the path to output the file
            output_file_path = os.path.join(noise_reduced_wav_path, relative_path)
            # Creating directories if they dont already exist
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            # Saving the noise reduced to file to another folder
            noise_reduced_wav.export(output_file_path, format="wav")
            
print('Noise Reduction Process Ended')
