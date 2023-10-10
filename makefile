env:
	brew install python-tk@3.9
	pip3.9 install mne
	pip3.9 install pandas
	pip3.9 install seaborn
	pip3.9 install scikit-learn

visualize_preprocessing:
	cd processing_EEGs_lib && python3.9 preprocessing.py -v
