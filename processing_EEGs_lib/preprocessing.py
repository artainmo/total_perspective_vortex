import sys
import mne
import matplotlib 
matplotlib.use('TkAgg') #To work on macos monterey
import matplotlib.pyplot as plt
from .libs.NeuralNetworkLib.manipulate_data import normalization_zscore
import numpy as np

# Download datasets
# Learn more about datasets (https://physionet.org/content/eegmmidb/1.0.0/, 
#           https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html#footcite-schalketal2004).
def get_data():
    # This returns a list of pathlib.PosixPath objects referring to 14 datasets each describing a different task
    data_paths = mne.datasets.eegbci.load_data(1, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], path="./datasets", 
                                force_update=False, update_path=True, 
                                base_url='https://physionet.org/files/eegmmidb/1.0.0/', verbose=None)
    #left_right_fist_datasets = [3, 4, 7, 8, 11, 12] #Use all datasets that describe the same for more training data
    left_right_fist_datasets = [4,7] #Use all datasets that describe the same for more training data
    raws = [mne.io.read_raw_edf(data_paths[number-1], preload=True) for number in left_right_fist_datasets]
    raw_data = mne.io.concatenate_raws(raws)
    annotations = raw_data.annotations.copy()
    return raw_data, annotations
# Returns mne.raw object containing the datas in a raw forrmat
# We will use dataset 3 where we will try to predict if someone is 
# at rest (T0), moves left fist (T1) or moves right fist (T2)
# It also returns the annotations which basically are the answers (T0, T1, T2)

def clean_annotations(annot):
    to_remove = []
    for i, a in enumerate(annot):
        if len(a['description']) > 2 and a['description'][0:3].lower() == "bad":
            to_remove.append(i)         
    annot.delete(to_remove)
# Remove all the annotations whereby a recording failed

# Visualize raw data
def visualize_raw(raw_data):
    raw_data.plot(title='Run 1 - Baseline eyes open', n_channels=len(raw_data.ch_names), block=True)
# Because of too much data to visualize, we are not able to detect homogeneity between channels

# Filter frequency bands
g_freq_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'gamma': (30, 79)
}
def filter_frequency_bands(raw_data, freq_bands = g_freq_bands):
    filtered_data = {}
    for band, (low_freq, high_freq) in freq_bands.items():
        filtered_data[band] = mne.filter.filter_data(raw_data.get_data(), raw_data.info['sfreq'], low_freq, high_freq)
        filtered_data[band] = filtered_data[band].T 
        filtered_data[band] = normalization_zscore(filtered_data[band]) 
    return filtered_data
# Returns dictionary of requested frequency bands
# Each frequency band key contains a numpy array representing the measurements over time (160 per second during 2min)
# each containing a numpy array with all those measurements in relation to the 64 channels/electrodes
# All values got normalized with z-score, allowing outliers

def annot_time_to_graph_time(annot):
    middle_time = annot['onset'] + annot['duration']/2
    return int(middle_time * 160) #160 measurements are made per second...

# Visualize filtered frequency bands data 
def visualize_filtered_frequency_bands_data(filtered_data, annotations):
    fig, axes = plt.subplots(len(filtered_data), 1)
    for i, key in enumerate(filtered_data.keys()):
        axes[i].plot(filtered_data[key])
        for annot in annotations:
            axes[i].text(annot_time_to_graph_time(annot), axes[i].get_ylim()[1], annot['description'], ha='center')
        axes[i].set_title(key)
    plt.tight_layout()
    plt.show()

def annot_time_to_rawData_time(time_in_sec):
    return int(time_in_sec * 160) #160 EEG measurements are made per second...

def frequency_bands_data_per_state(datas, annotations):
    ret = {}
    for key in datas.keys():
        for annot in annotations:
            start = annot_time_to_rawData_time(annot['onset'])
            end = annot_time_to_rawData_time(annot['onset'] + annot['duration'])
            if key not in ret:
                ret[key] = [datas[key][start:end].T]
            else:
                ret[key].append(datas[key][start:end].T)
    return ret
# Returns dict of frequency-bands each containing a 3D numpy array of each state in time 
# containing associated measurements of each channel/electrode.
# Thus it has shape (30, +-672, 64) with 30 referring to 30 states over the 2min time period, 
# +-672 measurements made over each +-4.1 seconds (time of each state) for each of the 64 channels.
# In the end using .T (transpose), the returned object has shape (30, 64, +-672) so that the recordings made in time
# are placed relative to each channel

def extract_power(data_per_state):
    if len(data_per_state.keys()) == 0:
        print("Error: no frequency bands")
        exit()
    nb_channels = data_per_state[list(data_per_state.keys())[0]][0].shape[0]
    nb_states = len(data_per_state[list(data_per_state.keys())[0]])
    nb_freq_bands = len(data_per_state.keys())
    ret = np.empty((0, nb_channels * nb_freq_bands))
    for row in range(nb_states):
        ret_row = np.array([])
        for channel in range(nb_channels):
            for freq in data_per_state.keys():
                fft = np.fft.fft(data_per_state[freq][row][channel]) # fast fourier transform
                # calculate the power spectral density (PSD) by taking the square of the magnitude of the FFT result
                psd = np.abs(fft) ** 2
                power = np.sum(psd) # addition all of them to one value to get total power
                ret_row = np.append(ret_row, power)
        ret = np.append(ret, [ret_row], axis=0)
    return ret
# Returns numpy array (30, 192)  with first dimension equal to number of examples and second dimension equal to
# power per frequency-band of each channel

def initial_cleaning(raw_data, annotations):
    # annotations.rename({'T0': 'BAD'}) # test cleaning
    clean_annotations(annotations)
    raw_data.drop_channels(raw_data.info['bads']) # Remove bad channels from the start
    raw_data.pick_types(eeg=True) # Only keep EEG channels

def ffb(filtered_frequency_bands_data): #filter_frequency_bands
    del filtered_frequency_bands_data['theta']
    del filtered_frequency_bands_data['beta']

def transform_to_x_values(filtered_frequency_bands_data, annotations):
    data_per_state = frequency_bands_data_per_state(filtered_frequency_bands_data, annotations)
    return extract_power(data_per_state)

def transform_to_y_values(annotations):
    ret = np.array([])
    for annot in annotations:
        ret = np.append(ret, [int(annot['description'][1])])
    return ret

def remove_class_labels(x, y):
    #annotation 0 equals rest and can be removed as we only want binary classification between two movements
    remove_indexes = np.where(y == 0)[0]
    y = np.delete(y, remove_indexes)
    x = np.delete(x, remove_indexes, axis=0)
    return x, y

def preprocessing_transformation(raw_data, annotations):
    print(raw_data.__len__)
    print(len(annotations))
    initial_cleaning(raw_data, annotations) 
    print(raw_data.__len__)
    print(len(annotations))
    filtered_frequency_bands_data = filter_frequency_bands(raw_data)
    print(filtered_frequency_bands_data['delta'].shape)
    print(len(annotations))
    ffb(filtered_frequency_bands_data)
    print(filtered_frequency_bands_data['delta'].shape)
    print(len(annotations))
    x_values = transform_to_x_values(filtered_frequency_bands_data, annotations)
    y_values = transform_to_y_values(annotations)
    return remove_class_labels(x_values, y_values)


if __name__ == "__main__":
    raw_data, annotations = get_data()
    initial_cleaning(raw_data, annotations)
    if len(sys.argv) > 1 and sys.argv[1] == "-v":
        visualize_raw(raw_data) 
    filtered_frequency_bands_data = filter_frequency_bands(raw_data)
    if len(sys.argv) > 1 and sys.argv[1] == "-v":
        visualize_filtered_frequency_bands_data(filtered_frequency_bands_data, annotations)
    # Here we can see that theta gives out similar signals as delta but at a lower amplitude, thus we can remove theta
    # Also we can see that beta gives out similar signals as alpha but at a lower amplitude, thus we can remove beta
    ffb(filtered_frequency_bands_data)
    if len(sys.argv) > 1 and sys.argv[1] == "-v":
        visualize_filtered_frequency_bands_data(filtered_frequency_bands_data, annotations)
    x_values = transform_to_x_values(filtered_frequency_bands_data, annotations)
    print(x_values.shape)
    

