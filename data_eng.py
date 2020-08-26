"""
With this experiment we have the restriction of needing to have unique patients in the test and the
train split. But we also have a limited amount of memory to load the coil data. In order for us to run a k=fold cross validation we need to be able to index the coil data on disk before retreiving it.
We use the h5 file storage format here to that end.

"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit
from pathlib import Path
import os
from utils import (
    HealthChecker,
    get_beats,
    load_example,
    get_patient,
)

data_folder = "/home/edent/Projects/Demos/ECG-Classification-CNN/data/physionet.org/files/ptbdb/1.0.0/"


@njit
def extract_heartbeats(data, beats, padding):
    beats = beats[beats > padding]
    beats = beats[beats < data.shape[0] - padding]
    out = np.empty((len(beats), padding*2+1, 15))
    for i, beat in enumerate(beats):
        add = data[beat-padding: beat+padding+1, :]
        try:
            out[i, :, :] = add
        except:
            print(add.shape, beat, data.shape[0] - padding)
    return out


def main():
    out_folder = "/home/edent/Projects/Demos/ECG-Classification-CNN/data/train/"
    paths = list(Path(data_folder).rglob('*.hea'))
    hc = HealthChecker(data_folder+'CONTROLS')

    PADDING = 500
    coil_data = []
    labels = []
    patients = []

    for path in tqdm(paths):
        sample_path = str(path.absolute())[:-4]
        data, _ = load_example(path)
        beats = get_beats(data, k=50)
        coil_beats = extract_heartbeats(data, beats, PADDING)
        coil_data.append(coil_beats)
        patients.append(np.full((coil_beats.shape[0],), get_patient(path)))
        labels.append(
            np.full((coil_beats.shape[0],), hc.healthy_sample(sample_path)))

    np.save(out_folder+'coil_data.npy', np.vstack(coil_data))
    np.save(out_folder+'patients.npy', np.hstack(patients))
    np.save(out_folder+'labels.npy', np.hstack(labels))


if __name__ == '__main__':
    main()
