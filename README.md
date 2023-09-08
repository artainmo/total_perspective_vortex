# total_perspective_vortex

42 school [subject](https://cdn.intra.42.fr/pdf/pdf/86321/en.subject.pdf).

Create a machine-learning pipeline that implements a dimensionality reduction algorithm to pre-process data before classification using scikit-learn.

## Preliminary Notions

### EEG datas

Brainwaves, from voltage fluctuations between neurons.

### ML Pipeline

ML workflows include data preparation, training and evaluation. ML pipelines automate the ML workflow and can be re-used on different datas to train or predict. Making the AI more reproducible.

### Real time data stream classification

Consists of a classifier that is trained in real time on incoming data and able to drift (adapt in real time), to stay accurate over time.

### MNE

Python package to handle EEG datas.

### scikit-learn

Python package for machine learning. In comparison to tensorflow it is higher level, while tensorflow is lower level, thus tensorflow allows for more customization and is often preferable for deep learning.

## ML pipeline structure

### Data processing

#### Preprocessing

Visualize the data and filter out features with no predictive power for the desired output. Most filtering algorithms use fourier transform or wavelet transform.

##### Signal spectrum

Signal spectrum is the range of frequencies found in a signal

##### Fourier transform

Is an integral transformation that decomposes signals into its constituent components and frequencies. It is capable of decomposing complicated waveforms into a sequence of simpler waves.
(Integral of a function is the inverse of the derivation of a function. While derivatives find slope of function, integral find volume under function curve.)

##### Wavelet transform

The Fourier transform captures global frequency information, this decomposition can generalize too much and be less precise depending on signals. Wavelet transform resolves this problem by allowing to extract local spectras.

Decomposes a function into wavelets. A wavelet is a wave-like oscillation that is localized in time, consists of two properties, scale and location. 

#### Processing 

##### Dimensionality reduction algorithm: 

Is an unsupervised learning technique. That can also be used to pre-process data for supervised learning after cleaning the data and normalizing it. Different dimensionality reduction algorithms exist, no method is superior it depends on dataset and must be tested. They have to be fit and evaluated as well on training and test set.

Dimensionality reduction seeks a lower-dimensional representation of numerical input data that preserves the salient relationships in the data. This allows for a reduced number of parameters in your supervised learning model, simpler models are desirable.

In simple terms, it reduces the mumber of input variables in a predictive model. This leads to improved performance of predictive model.

###### PCA

Principal component analysis, most popular technique in dimensionality reduction, used to preprocess data, comes from linear algebra.

# Resources

[6 Dimensionality Reduction Algorithms With Python](https://machinelearningmastery.com/dimensionality-reduction-algorithms-with-python)
