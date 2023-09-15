import sys
import mne
import matplotlib 
matplotlib.use('TkAgg') #To work on macos monterey
import matplotlib.pyplot as plt

# Download datasets
# Learn more about datasets (https://physionet.org/content/eegmmidb/1.0.0/, 
#           https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html#footcite-schalketal2004).
def get_data():
    data_paths = mne.datasets.eegbci.load_data(1, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], path="./datasets", 
                                force_update=False, update_path=True, 
                                base_url='https://physionet.org/files/eegmmidb/1.0.0/', verbose=None)
    # This returns a list of pathlib.PosixPath objects referring to 14 datasets each describing a different task
    return mne.io.read_raw_edf(data_paths[2], preload=True)
# Returns mne.raw object containing the datas in a raw format
# We will use dataset 3 where will try to predict if someone is 
# at rest (T0), moves left fist (T1) or moves right fist (T2)

# Visualize raw data
def visualize_raw(raw_data):
    raw_data.plot(title='Run 1 - Baseline eyes open', n_channels=len(raw_data.ch_names), block=True)

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
    return filtered_data
# Returns dictionary of requested frequency bands
# Each frequency band key contains a numpy array representing the 64 channels, 
# each containing a numpy array with all the measurements over time (160 per second during 2min)

# Visualize filtered frequency bands data 
def visualize_filtered_frequency_bands_data(filtered_data):
    fig, axes = plt.subplots(len(filtered_data), 1)
    for i, key in enumerate(filtered_data.keys()):
        axes[i].plot(filtered_data[key])
        axes[i].set_title(key)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    raw_data = get_data()
    if len(sys.argv) > 1 and sys.argv[1] == "-v":
        visualize_raw(raw_data)
    filtered_frequency_bands_data = filter_frequency_bands(raw_data)
    if len(sys.argv) > 1 and sys.argv[1] == "-v":
        visualize_filtered_frequency_bands_data(filtered_frequency_bands_data)
    # Here we can see that theta gives out similar signals as delta but at a lower amplitude, thus we can remove theta
    # Also we can see that beta gives out similar signals as alpha but at a lower amplitude, thus we can remove beta
    del filtered_frequency_bands_data['theta']
    del filtered_frequency_bands_data['beta']
    if len(sys.argv) > 1 and sys.argv[1] == "-v":
        visualize_filtered_frequency_bands_data(filtered_frequency_bands_data)
    print(len(filtered_frequency_bands_data['alpha'][0]))

