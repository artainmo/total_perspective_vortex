# total_perspective_vortex

42 school [subject](https://cdn.intra.42.fr/pdf/pdf/86321/en.subject.pdf).

Process EEG datas by cleaning, extracting, creating a ML pipeline implementing a dimensionality reduction algorithm before classifying and handling a real time data-stream with sklearn.

## Preliminary Notions

### EEG datas

Brainwaves, from voltage fluctuations between neurons.

### ML Pipeline

ML workflows include data preparation, training and evaluation. ML pipelines automate the ML workflow and can be re-used on different datas to train or predict. Making the AI more reproducible.

### Real time data stream classification

Consists of a classifier that is trained in real time on incoming data and able to drift (adapt in real time), to stay accurate over time.

### MNE

Python package to handle EEG datas.

### PhysioNet
[PhysioNet](https://physionet.org/) is an online repository of freely-available medical research data, managed by the MIT Laboratory for Computational Physiology.<br>
We will use a dataset of EEGs related to motor movements, found and explained [here](https://physionet.org/content/eegmmidb/1.0.0/).

### scikit-learn

Python package for machine learning also called sklearn. In comparison to tensorflow it is higher level, while tensorflow is lower level, thus tensorflow allows for more customization and is often preferable for deep learning.

## ML pipeline structure

### Data processing

#### Preprocessing
Parse the data. In this case for EEG use the specialized python library called [MNE](#MNE).<br>

Visualize the data and remove features with no predictive power for the desired output. Basically do [general data preparation](https://github.com/artainmo/neural-networks/tree/main#data-preparation--visualization).

Extract new features from the data.<br>
For example in the case of EEGs use the power of the signal by frequency and channels/electrodes.<br>
Or use the [signal's spectrum](#Signal-spectrum) and find it using an algorithm like [fourier transform](#Fourier-transform) or [wavelet transform](#Wavelet-transform).

##### Signal spectrum

Signal spectrum is the range of frequencies found in a signal.

##### Fourier transform

Is an integral transformation that decomposes signals into its constituent components and frequencies. It is capable of decomposing complicated waveforms into a sequence of simpler waves.<br>
Integral of a function is the inverse of the derivation of a function. While derivatives find slope of function, integral find volume under function curve.

##### Wavelet transform

The Fourier transform captures global frequency information, this decomposition can generalize too much and be less precise depending on signals. Wavelet transform resolves this problem by allowing to extract local spectras.<br>
Wavelet transform decomposes a function into wavelets. A wavelet is a wave-like oscillation that is localized in time, consists of two properties, scale and location. 

#### Processing 

##### Dimensionality reduction algorithm: 

Is an unsupervised learning technique. That can also be used to pre-process data for supervised learning after cleaning the data and normalizing it.<br>
Different dimensionality reduction algorithms exist, no method is superior, it depends on dataset and must be tested. They have to be used on both training and test set.

Dimensionality reduction seeks a lower-dimensional representation of numerical input data that preserves the salient relationships in the data. This allows for a reduced number of parameters in your supervised learning model.<br>
In simple terms, it reduces the number of input variables in a predictive model by eliminating input variables with no predictive power. This leads to lower computational cost and improved performance of the predictive model.

###### PCA

Principal component analysis is the most popular technique in dimensionality reduction when data is dense, meaning the data contains few zero values.<br>
It uses simple matrix operations from linear algebra and statistics to lower the dimensionality of the dataset.

###### SVD

Singular value decomposition is most often used when data is sparse, meaning it contains many zero values.<br>
This dimensionality reduction method uses matrix decomposition also known as matrix factorization which reduces a matrix to its constituent parts.

###### CSP

Common spatial patterns is the most used dimensionality reduction algorithm when handling EEG datas.<br>
It is used in signal processing, such as EEG signals, to find the most salient patterns in the data thus putting a spotlight on the input datas with most predictive value.

### Model training

To train the model first [split the data](https://github.com/artainmo/neural-networks/tree/main#data-splitting) into a training, validation and test set.<br>
Research and find the best sklearn classification algorithm for this project. 

### Deployment

The final stage is applying the ML model to the production area allowing users to get pedictions on live data.<br>
Here, [real time data stream classification](#Real-time-data-stream-classification) can also be applied so that the model can keep learning in real time.

# Resources

[6 Dimensionality Reduction Algorithms With Python](https://machinelearningmastery.com/dimensionality-reduction-algorithms-with-python)<br>
[How to Calculate Principal Component Analysis (PCA) from Scratch in Python](https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/)<br>
[How to Calculate the SVD from Scratch with Python](https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/)<br>
[Machine Learning Pipeline: Architecture of ML Platform in Production](https://www.altexsoft.com/blog/machine-learning-pipeline/)<br>


