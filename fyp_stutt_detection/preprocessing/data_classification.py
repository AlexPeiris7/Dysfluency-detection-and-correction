import os
import pathlib
import shutil
import pandas as pd

csv_labels = '/Users/alexiopeiris/Desktop/University/FourthYear/FYP/Imp/Stuttering_detection_and_correction/stuttering_detection/preprocessing/SEP-28k/SEP-28k_labels.csv'
spectrograms_dir = '/Users/alexiopeiris/Desktop/University/FourthYear/FYP/Imp/Stuttering_detection_and_correction/stuttering_detection/preprocessing/spectrograms'
clsf_dtst_dir = '/Users/alexiopeiris/Desktop/University/FourthYear/FYP/Imp/Stuttering_detection_and_correction/stuttering_detection/preprocessing/classified_spectrograms'

# Load csv label file
data = pd.read_csv(csv_labels, dtype={"EpId": str})

# store labels from csv file
shows = data.Show
ep_id = data.EpId
clip_id = data.ClipId
starts = data.Start
stops = data.Stop
labels = data.iloc[:, 5:].values
prolongations = data.Prolongation
blocks = data.Block
sound_reps = data.SoundRep
word_reps = data.WordRep
interjections = data.Interjection

n_items = len(shows)

for i in range(n_items):
    clip_idx = clip_id[i]
    show_abrev = shows[i]
    episode = ep_id[i].strip()

    if prolongations[i] != 0 or blocks[i] != 0 or sound_reps[i] != 0 or word_reps[i] != 0 or interjections[i] != 0:
        stuttering_present = True
    else:
        stuttering_present = False

    # spectrogram path
    # Using formatted literal string to concatenate
    spec_path = f"{spectrograms_dir}/{shows[i]}_{episode}_{clip_idx}_spec.png"

    if stuttering_present:
        dataset_type = "stuttering"
    else:
        dataset_type = "no_stuttering"

    # Create dir if it does not exist
    os.makedirs(clsf_dtst_dir, exist_ok=True)

    # classifying the dataset into 2 types, dataset with stuttering and dataset without stuttering
    # resetting original dataset folder structure as it was not possible in the extract spectrogram
    # due to the error with matplotlib
    dataset_dir = pathlib.Path(clsf_dtst_dir + '/' + dataset_type + '/')
    # Using formatted literal string to concatenate
    classified_dataset_path = f"{dataset_dir}/{shows[i]}_{episode}_{clip_idx}.png"

    if not os.path.exists(spec_path):
        print("Missing", spec_path)
        continue

    # Create dir if it does not exist
    os.makedirs(dataset_dir, exist_ok=True)

    # Copy spectrogram into dataset_types folders to classify
    shutil.copy(spec_path, classified_dataset_path)
