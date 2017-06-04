# Classification of Interferometer States



### Ideas

1. What is the seismic state? (e.g. EQ, truck, windy, broken sensor)
1. Are the optical levers going bad? (e.g. look at the noise at 1 Hz vs. the noise at 50 Hz)
1. Do we want to use Clustering or Classification? (probably clustering because of [Star Wars](http://stackoverflow.com/questions/5064928/difference-between-classification-and-clustering-in-data-mining))
1. In The End, after we have clustering results, we would like to go through and label all of the clusters so that this can form the basis of future analyses where we just do Classification into these previously discovered CLusters.



### Classification Papers/Algorithms
1. [Unsupervised Feature Selection for Pattern Search in Seismic Time Series](http://www.jmlr.org/proceedings/papers/v4/koehler08a/koehler08a.pdf): seems like they first use statistical tests to find significant features and then use Self-Organizing Maps on these features

2. [Synopsis of supervised and unsupervised pattern classification techniques](http://adsabs.harvard.edu/full/2009GeoJI.178.1132L) applied to volcanic tremor data at Mt Etna, Italy: for the unsupervised methods, used Cluster Analysis (CA) and SOM

3. Clustering of time series data—a survey by T. Warren Liao: Clustering methods can be divided into five types, of which partitioning and model-based methods seem the most relevant
  <ul>
  <li>Partitioning: construct k partitions/clusters of the data; crisp partitioning classifies each object into exactly one     
  cluster, while fuzzy clustering allows an object to be in more than one cluster to a different degree
  <ol>
    <li> k-means clustering: crisp clustering where each cluster is represented as the mean of all objects in the cluster</li>
    <li>fuzzy c-means algorithm: fuzzy analog to k-means</li>
  </ol>
  </li>
  <li>Model-based methods: statistical approach or neural network approach
  <ol>
  <li>Statistical approaches - Bayesian statistical analysis to estimate the number of clusters</li>
  <li>Neural network methods - competitive learning, including ART (adaptive resonance theory) and self-organizing maps</li>
  </ol>
  </li>
  </ul>
</p>

4. [Pattern Recognition in Time Series](https://pdfs.semanticscholar.org/2f5a/4b8b158117928e9eee7ac6ce7da291ec9bd2.pdf): common unsupervised learning algorithhms for time-series classification includes 
  * k-means, hierarchial clustering (limited to small datasets because of its complexity), 
  * EM (Expectation-Maximization). EM is similar to k-means, but models every data object as having some degree of membership to its cluster and can model many moe cluster shapes, whereas k-means assigns each data object to only one clsuter and generates spherical cluster. IMPORTANT NOTE - quality of clusters in k-means depends on initial cluster center picks

----
### Input Data Format

Each column row is a time point, with 12 features corresponding to the following 12 channels:
```
C1:PEM-RMS_BS_X_0p03_0p1
C1:PEM-RMS_BS_X_0p1_0p3
C1:PEM-RMS_BS_X_0p3_1
C1:PEM-RMS_BS_X_10_30
C1:PEM-RMS_BS_X_1_3
C1:PEM-RMS_BS_X_3_10
C1:PEM-RMS_BS_Y_0p03_0p1
C1:PEM-RMS_BS_Y_0p1_0p3
C1:PEM-RMS_BS_Y_0p3_1
C1:PEM-RMS_BS_Y_10_30
C1:PEM-RMS_BS_Y_1_3
C1:PEM-RMS_BS_Y_3_10
C1:PEM-RMS_BS_Z_0p03_0p1
C1:PEM-RMS_BS_Z_0p1_0p3
C1:PEM-RMS_BS_Z_0p3_1
C1:PEM-RMS_BS_Z_10_30
C1:PEM-RMS_BS_Z_1_3
C1:PEM-RMS_BS_Z_3_10
```
(basically, X and Y direction readings across all the frequency bands)
The Python script to access and compile data into this format can be found as 'getData.py'. 



### K-Means Clustering
I chose k-means clustering because it seemed to fit what we were looking for. The MATLAB code for creating classifying through k-means and generating plots for these classifications can be found as clusterPlotting.m
The classification needs to be modified by finding the optimal number of clusters(see below) and the model needs to be evaluated somehow. 


### Optimizing Number of Clusters in K-Means
Plot the mean distance to a centroid as a function of k - the number of clusters - and use the "elbow point" to estimate the optimal number of clusters, as shown in https://www.datascience.com/blog/introduction-to-k-means-clustering-algorithm-learn-data-science-tutorials

### Evaluation


### Classification Possibilities
<p> Experiment in Matlab with: </p>
<ol>
<li><em>Self-organizing map - unsupervised neural network to find clusters within dataset</em></li>
<li><em>k-means clustering</em></li>
</ol>

## Clustering in python:
1. http://scikit-learn.org/stable/modules/clustering.html
------

## Visualization of Clustering

1. http://www.dummies.com/programming/big-data/data-science/how-to-visualize-the-clusters-in-a-k-means-unsupervised-learning-model/
1. http://www.informationisbeautiful.net/
1. http://colah.github.io/posts/2014-10-Visualizing-MNIST/
------

### Getting Data
1. ![GWpy](GWpyReadme.md)
1. getData.py
1. getMinuteTrend.py
1. 40m Dataviewer download



![ML Cheat Sheet](microsoft-machine-learning-algorithm-cheat-sheet-v6.png?raw=true "Title")

------
[Markdown Synax - GitHub flavor](https://help.github.com/articles/basic-writing-and-formatting-syntax/)
