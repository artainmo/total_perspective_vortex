import sys
import mne
import matplotlib 
matplotlib.use('TkAgg') #To work on macos monterey
import matplotlib.pyplot as plt
from libs.NeuralNetworkLib.manipulate_data import normalization_zscore

# Download datasets
# Learn more about datasets (https://physionet.org/content/eegmmidb/1.0.0/, 
#           https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html#footcite-schalketal2004).
def get_data():
    # This returns a list of pathlib.PosixPath objects referring to 14 datasets each describing a different task
    data_paths = mne.datasets.eegbci.load_data(1, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], path="./datasets", 
                                force_update=False, update_path=True, 
                                base_url='https://physionet.org/files/eegmmidb/1.0.0/', verbose=None)
    raw_data = mne.io.read_raw_edf(data_paths[2], preload=True)
    annotations = raw_data.annotations.copy()
    return raw_data, annotations
# Returns mne.raw object containing the datas in a raw format
# We will use dataset 3 where we will try to predict if someone is 
# at rest (T0), moves left fist (T1) or moves right fist (T2)
# It also returns the annotations which basically are the answers (T0, T1, T2)

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
    'gamma': (30, 40)
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

if __name__ == "__main__":
    raw_data, annotations = get_data()
    if len(sys.argv) > 1 and sys.argv[1] == "-v":
        visualize_raw(raw_data) 
    filtered_frequency_bands_data = filter_frequency_bands(raw_data)
    if len(sys.argv) > 1 and sys.argv[1] == "-v":
        visualize_filtered_frequency_bands_data(filtered_frequency_bands_data, annotations)
    # Here we can see that theta gives out similar signals as delta but at a lower amplitude, thus we can remove theta
    # Also we can see that beta gives out similar signals as alpha but at a lower amplitude, thus we can remove beta
    del filtered_frequency_bands_data['theta']
    del filtered_frequency_bands_data['beta']
    if len(sys.argv) > 1 and sys.argv[1] == "-v":
        visualize_filtered_frequency_bands_data(filtered_frequency_bands_data, annotations)
    for annotation in annotations:
        print(annotation)

