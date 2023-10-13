# total_perspective_vortex

42 school [subject](https://cdn.intra.42.fr/pdf/pdf/86321/en.subject.pdf).

Process EEG datas by cleaning, extracting, creating a ML pipeline implementing a dimensionality reduction algorithm before classifying and handling a real time data-stream with sklearn.

Learn more by reading this README and expanded comments in code.

## Use

Compile with **python3.9**.

Optionally use flag **-v** for plot visualization.

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
To do that find the [signal spectrum](#Signal-spectrum) using an algorithm like [fourier transform](#Fourier-transform) or [wavelet transform](#Wavelet-transform).

##### Signal spectrum

The signal spectrum describes a signal's magnitude and phase characteristics as a function of frequency.

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

Research and find the best sklearn classification algorithm for this project.<br>
The choice of the best algorithm will depend on the EEG dataset, factors such as the number of classes, the dataset size, presence of outliers...<br>
It's a good practice to perform model selection and hyperparameter tuning using cross-validation to determine which algorithm performs best.<br>
Additionally, factors like interpretability, computational resources, and the need for real-time processing can be considered.

The following are classification models often used in the context of EEGs processed by CSP.<br>
I need to find one that works best for binary classification on small datasets while being fast and able to self-update (bonus) for real-time processing.

###### Linear Discriminant Analysis (LDA)
Finds a linear combination of features that characterizes or separates two or more classes of objects or events.<br>
It allows both binary classification and multi-class classification.<br>

Linear Discriminant Analysis (LDA) is generally effective on small datasets, and it is particularly well-suited for small sample size problems. LDA is a dimensionality reduction and classification technique that works by maximizing the separability of different classes while minimizing the variance within each class. It's particularly effective when the number of samples is much smaller than the number of features, which is a common characteristic of small datasets.

LDA is computationally efficient, especially when compared to more complex machine learning algorithms like deep neural networks. This makes it suitable for real-time applications where low latency is critical such as in the case of real-time processing.<br>
However if the application needs to adapt to changing data or classes over time, you'll need a mechanism to retrain the LDA model periodically.

###### Support Vector Machine (SVM)
SVMs are a set of supervised learning methods used for classification, regression and outliers detection.<br>
SVM's advantage is ability to perform well in high dimensional spaces, thus when dataset contains a lot of features.

###### Random Forest (RF) (and Decision trees)
A RF is a classifier that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.<br>
A decision tree is a decision support hierarchical model that uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is a supervised learning algorithm only made out of conditional control statements.

Decision trees can be highly efficient on small datasets because they are highly interpretable and computationally efficient which is a positive for real-time processing. If speed is important they are fast.<br>
Decision trees are particularly effective in low-dimensional feature spaces, where they can quickly identify decision boundaries.<br>
However decision trees may not capture complex relationships as effectively as more sophisticated algorithms like random forests, gradient boosting, or neural networks and decision trees can be sensitive to variations in the data which small datasets are more susceptible to.

Random Forests rely on bootstrapping (randomly sampling the data with replacement) to create multiple trees. In small datasets, the randomness from bootstrapping can lead to high variability between different runs. This can be addressed by increasing the number of trees in the forest. Random forests can also be computationally expensive on small datasets when a lot of features are present. Lastly, feature importance estimates in Random Forests may be biased on small datasets.

Both RF and decision trees don't necessitate normalization of data and are not sensitive to outliers.

###### Neural Networks
Deep learning models like Convolutional Neural Networks (CNNs) or Recurrent Neural Networks (RNNs) can also be applied to EEG data. CNNs can be used to capture spatial features, and RNNs can be applied to time-series EEG data. However, they typically require larger amounts of data and computational resources.

Stochastic-gradient-descend, a technique used in neural-networks, is ideal for incremental self-updating of the model during real-time processing.

###### K-Nearest Neighbors (KNN)
KNN is a simple and interpretable supervised classification algorithm that can be used with CSP-transformed EEG data.

It relies on the idea that similar data points tend to have similar labels or values.<br>
During the training phase, the KNN algorithm stores the entire training dataset as a reference. When making predictions, it calculates the distance between the input data point and all the training examples, using a chosen distance metric such as Euclidean distance.<br>
Next, the algorithm identifies the K nearest neighbors to the input data point based on their distances. In the case of classification, the algorithm assigns the most common class label among the K neighbors as the predicted label for the input data point. For regression, it calculates the average or weighted average of the target values of the K neighbors to predict the value for the input data point.

KNN is often considered a good choice for small datasets because it is easy to setup and effective on low-dimensional datasets.<br>
KNN can capture non-linear relationships in the data. If the small dataset exhibits complex patterns, KNN can adapt to it.
Its training is very fast as it only needs to store the training samples. Which is good for real-time processing and subsequent self-updating. Some variations of KNN, such as the Large Margin Nearest Neighbors (LMNN) algorithm, support incremental learning by updating the nearest neighbor search structure as new data arrives.<br>
However when the dataset is large it is computationally expensive and thus slow. Also when dataset is large KNN becomes memory expensive by keeping a copy of it in memory.<br>
Thus overall KNN is best for small datasets.

The choice of the number of neighbors (K) is critical. A small K may lead to overfitting, while a large K may lead to underfitting. A larger K also makes the algorithm slower. <br>
KNN can be sensitive to outliers and noise in the data. It's important to preprocess the data to remove or mitigate outliers, thus use min-max normalization. A higher K can also mitigate the perturbance of outliers.

###### Gradient Boosting
Gradient Boosting is a powerful boosting algorithm that combines several weak learners into strong learners, in which each new model is trained to minimize the loss function such as mean squared error or cross-entropy of the previous model using gradient descent. In each iteration, the algorithm computes the gradient of the loss function with respect to the predictions of the current ensemble and then trains a new weak model to minimize this gradient. The predictions of the new model are then added to the ensemble, and the process is repeated until a stopping criterion is met.

Gradient Boosting algorithms include XGBoost, LightGBM, and CatBoost.

Gradient Boosting is known for its high predictive performance and the ability to capture complex patterns in data. This makes it effective when accuracy is a top priority. And this even on small datasets.<br>
Gradient Boosting can be computationally intensive, particularly when you have many weak learners (trees) in the ensemble. This may not be an issue on small datasets, but it's something to keep in mind as dataset sizes grow.<br>
Hyperparameter tuning becomes more critical on small datasets, as there is less data to learn from.<br>

Some implementations of Gradient Boosting, such as LightGBM and CatBoost, are designed to be efficient and can provide fast predictions. They utilize techniques like histogram-based gradient boosting and optimized data structures to speed up processing.<br>
Gradient Boosting algorithms can be computationally intensive, and memory usage can be substantial. In real-time processing, this may be a concern, particularly for very large datasets.<br>
Lacks built-in mechanism for incremental update to the model.

### Deployment

The final stage is applying the ML model to the production area allowing users to get pedictions on live data.<br>
Here, [real time data stream classification](#Real-time-data-stream-classification) can also be applied so that the model can keep learning in real time.

# Resources

[machinelearningmastery - 6 Dimensionality Reduction Algorithms With Python](https://machinelearningmastery.com/dimensionality-reduction-algorithms-with-python)<br>
[machinelearningmastery - How to Calculate Principal Component Analysis (PCA) from Scratch in Python](https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/)<br>
[machinelearningmastery - How to Calculate the SVD from Scratch with Python](https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/)<br>
[Machine Learning Pipeline: Architecture of ML Platform in Production](https://www.altexsoft.com/blog/machine-learning-pipeline/)<br>
[stackoverflow - What is exactly sklearn.pipeline.Pipeline?](https://stackoverflow.com/questions/33091376/what-is-exactly-sklearn-pipeline-pipeline)<br>
[machinelearningmastery - Modeling Pipeline Optimization With scikit-learn](https://machinelearningmastery.com/modeling-pipeline-optimization-with-scikit-learn/)<br>
[stackoverflow - Why should we use BaseEstimator and Transformermixmin from sklearn.base in python to create/ custom class?](https://stackoverflow.com/questions/72729841/why-should-we-use-baseestimator-and-transformermixmin-from-sklearn-base-in-pytho)<br>
[geeksforgeeks - What is the difference between ‘transform’ and ‘fit_transform’ in sklearn-Python?](https://www.geeksforgeeks.org/what-is-the-difference-between-transform-and-fit_transform-in-sklearn-python/)<br>
[Creating custom scikit-learn Transformers](https://www.andrewvillazon.com/custom-scikit-learn-transformers/)<br>

