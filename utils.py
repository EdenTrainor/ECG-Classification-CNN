import numpy as np
import pandas as pd
import wfdb
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from numba import njit


def load_example(path):
    p = str(path.absolute())[:-4]
    data, info = wfdb.rdsamp(p)
    return data, info


def plot_array(data, fig=None, c=None, title=None):
    """
    Plots a 15 channel ecg sample with labels.
    """
    sig_names = ['i', 'ii', 'iii', 'avr', 'avl', 'avf',
                 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'vx', 'vy', 'vz']
    if not fig:
        fig, axes = plt.subplots(nrows=15, figsize=(16, 7), sharex=True)
    else:
        axes = fig.axes
    for i, ax in enumerate(axes):
        ax.plot(range(data.shape[0]), data[..., i], label=sig_names[i], c=c)
        ax.legend(loc='upper right')
        ax.margins(x=0)
        if i != 0:
            ax.spines['top'].set_visible(False)
        if i != 14:
            ax.spines['bottom'].set_visible(False)
    if title:
        axes[0].set_title(title)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    return fig


class HealthChecker:
    """
    Loads the file containing all the control samples.
    Checks the path of samples passed against list of control patients.
    """

    def __init__(self, control_file):
        with open(control_file, 'r') as cf:
            self.controls = tuple(cf.read().splitlines())
        self.control_patients = {int(samp[7:10]) for samp in self.controls}

    def healthy_sample(self, path):
        sample_path = str(path.absolute())[:-4]
        if sample_path.endswith('.hea'):
            sample_path = str(sample_path.absolute())[:-4]
        if sample_path.endswith(self.controls):
            return 1
        return 0
    
    def healthy_patient(self, p_number):
        return int(p_number in self.control_patients)
    

def flatten(data):
    """

    """
    out = data.copy()
    kernel = np.ones(data.shape[0]//100)
    for i in range(data.shape[1]):
        out[:, i] -= np.convolve(data[:, i], kernel, mode='same')/len(kernel)
    return out


@njit
def normalize(data):
    for i in range(data.shape[1]):
        data[:, i] /= np.max(data[:, i]) - np.min(data[:, i])
    return data


def show_beats(data, k=50):
    
    #Make all beats equal
    data = flatten(data)
    data = normalize(data)
    
    #Highlight steep slopes
    data = np.gradient(data, axis=0)
    #Combine Channels
    data = np.mean(data, axis=1)
    #We don't care if its' up or down as long as it's steep
    data = np.abs(data)
    #Smooth over the peaks so the center is more pronounced
    data = np.convolve(data, np.ones(k), mode='same')
    return data

def isolate_beats(array):
    height = np.ptp(array)/3
    return find_peaks(array, height=height, distance=100)[0]


def get_beats(data, k=50):
    beats = show_beats(data, k)
    return isolate_beats(beats)


def plot_beats(plot_data):
    fig = plot_array(plot_data)
    peaks = get_beats(plot_data)
    for i, ax in enumerate(fig.axes):
        ymin, ymax = ax.get_ylim()
        for peak in peaks:
            ax.vlines(peak, ymin, ymax, color='r')
    return fig


def get_patient(path):
    return int(str(path.parent)[-3:])


def get_patient_list(paths):
    return sorted(list(set([get_patient(path) for path in paths])))

def get_n_samples(p_numbers):
    out = []
    for pnum in p_numbers:

def load_patients(p_numbers):
    coil_dir = ''
    sample_lens = pickle.load(sample_lens)
    n_samples = get_n_samples(p_numbers)
    layout = h5py.VirtualLayout((n_samples, 1000, 15), ...)
    
    
    