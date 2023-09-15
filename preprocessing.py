import mne
import matplotlib
matplotlib.use('TkAgg') #To work on macos monterey

# Download datasets
# Learn more about datasets (https://physionet.org/content/eegmmidb/1.0.0/, 
#           https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html#footcite-schalketal2004).
data_paths = mne.datasets.eegbci.load_data(1, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], path="./datasets", 
                            force_update=False, update_path=True, 
                            base_url='https://physionet.org/files/eegmmidb/1.0.0/', verbose=None)
# This returns a list of pathlib.PosixPath objects referring to 14 datasets each describing a different task

# Visualize raw data
raw_data = mne.io.read_raw_edf(data_paths[0], preload=True)
raw_data.plot(title='Run 1 - Baseline eyes open', n_channels=len(raw_data.ch_names), block=True)

# Filter frequency bands related to sleep
filtered_data = {}
freq_bands = {
    #'delta': (0.5, 4),
    #'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'gamma': (30, 40)
}
for band, (low_freq, high_freq) in freq_bands.items():
    filtered_data[band] = mne.filter.filter_data(raw_data.get_data(), raw_data.info['sfreq'], low_freq, high_freq)

# Visualize filtered data
for band, data in filtered_data.items():
    data.plot(title="Filter frequency bands: "+band, n_channels=len(data.ch_names), block=True)
 
