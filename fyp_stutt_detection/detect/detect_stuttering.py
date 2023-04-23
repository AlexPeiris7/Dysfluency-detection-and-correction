from pydub import AudioSegment
from pydub.utils import make_chunks
from scipy.io import wavfile
import matplotlib.pyplot as plot
import os
import numpy as np
from keras.models import load_model
import tensorflow as tf
import noisereduce as nr


# loading the model
model = tf.keras.models.load_model('/Users/alexiopeiris/Desktop/University/FourthYear/FYP/Imp/Stuttering_detection_and_correction/stuttering_detection/model/CRNN_model.h5', compile = False)
METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'),
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc')
]

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4), loss = 'binary_crossentropy', metrics = METRICS)

# assign files
mp3_file = "stuttered_speech.mp3"
wav_file = "input_wav.wav"

# convert mp3 file to wav file
input = AudioSegment.from_mp3(mp3_file)
input.export(wav_file, format="wav")

# loading file
input_audio = AudioSegment.from_file("input_wav.wav", "wav")
# converting the wav file in wav format to a numpy array
samples = input_audio.get_array_of_samples()
# noise removal process
# noise reduction using the imported module
reduce_noise = nr.reduce_noise(samples, sr=input_audio.frame_rate)
# creating new file to save the noise reduced audio file
noise_reduced_wav = AudioSegment(
    reduce_noise.tobytes(),
    frame_rate=input_audio.frame_rate,
    sample_width=input_audio.sample_width,
    channels=input_audio.channels)
# Saving the noise reduced to file to another folder
noise_reduced_wav.export('noise_removed_input.wav', format="wav")

# loading file
input_audio = AudioSegment.from_file("noise_removed_input.wav", "wav")
# splitting stereo file to mono (two channels to one channel)
mono_audios = input_audio.split_to_mono()
# only using one mono file and ignoring the other
mono_left = mono_audios[0].export("input_mono_left_file.wav", format="wav")
# mono_right = mono_audios[1].export("mono_right.wav", format="wav")

input_mono_audio = AudioSegment.from_file("input_mono_left_file.wav", "wav")

# lenght in ms
clip_length = 3000
# splitting files
splits = make_chunks(input_mono_audio, clip_length) #Make chunks of one sec
# creating dir to store the spectrograms to detect
current_directory = os.getcwd()
new_directory = os.path.join(current_directory, r'input_spectrograms')
if not os.path.exists(new_directory):
   os.makedirs(new_directory)

# exporting the clipped files
for i, split in enumerate(splits):
    clipped_file_name = "clipped{0}.wav".format(i)
    # exporting file
    print("exporting", clipped_file_name)
    split.export(clipped_file_name, format="wav")
    sample_rate, samples = wavfile.read(clipped_file_name)
    # ValueError: operands could not be broadcast together with shapes(256, 2, 1124)(256, 1)
    # error appears as the wav file has two channels(stereo audio)
    # need to split the two channel audio file to one channel audio(stereo to mono)
    spectrogram, frequency, times, image_handle = plot.specgram(samples, Fs=sample_rate)
    plot.savefig('input_spectrograms/' + clipped_file_name + '_spec.png')
    # plot.savefig('input_spectrograms/' + 'clipped1.wav' + '_spec.png')
    # deleting split audio file as there is no use
    print("deleting", clipped_file_name)
    os.remove(clipped_file_name)
    plot.close()
    print(spectrogram)
    predictions = model.predict(spectrogram)
    for i, prediction in enumerate(predictions):
        predicted_class = np.argmax(prediction)
        print(f"Clip {i+1}: Predicted class is {predicted_class}")

#
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.io import wavfile
# from keras.models import load_model
#
# model = load_model('CRNN_model.h5')
#
# input_audio = 'path/to/audio/file.wav'
# sample_rate, sample = wavfile.read(input_audio)
#
# # Trim the audio into 3-second clips
# clip_duration = 3 # seconds
# clip_length = clip_duration * sample_rate
# num_clips = len(sample) // clip_length
# audio_clips = np.array([sample[i*clip_length:(i+1)*clip_length] for i in range(num_clips)])
#
# # Preprocess the audio data into spectrograms
# spectrograms = np.array([librosa.feature.melspectrogram(y=clip, sr=sample_rate, n_mels=128) for clip in audio_clips])
# spectrograms = np.array([librosa.power_to_db(spectrogram, ref=np.max) for spectrogram in spectrograms])
# spectrograms = np.expand_dims(spectrograms, axis=-1)
#
# # Visualize the spectrograms of the audio file
# plt.figure(figsize=(10, 5))
# for i in range(num_clips):
#     plt.subplot(num_clips, 1, i+1)
#     plt.title(f"Clip {i+1} spectrogram")
#     plt.imshow(spectrograms[i].squeeze().T, aspect='auto', origin='lower')
# plt.tight_layout()
# plt.show()
#
# # Make predictions using the model
# predictions = model.predict(spectrograms)
#
# # Print the predicted class for each clip
# for i, prediction in enumerate(predictions):
#     predicted_class = np.argmax(prediction)
#     print(f"Clip {i+1}: Predicted class is {predicted_class}")
