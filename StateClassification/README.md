# Classification of Interferometer States



### Ideas

1. What is the seismic state? (e.g. EQ, truck, windy, broken sensor)
1. Are the optical levers going bad? (e.g. look at the noise at 1 Hz vs. the noise at 50 Hz)

![ML Cheat Sheet](microsoft-machine-learning-algorithm-cheat-sheet-v6.png?raw=true "Title")


### Classification Papers/Algorithms
<p>http://www.jmlr.org/proceedings/papers/v4/koehler08a/koehler08a.pdf - Unsupervised Feature Selection for Pattern Search in Seismic Time Series: seems like they first use statistical tests to find significant features and then use Self-Organizing Maps on these features</p>
<p>http://adsabs.harvard.edu/full/2009GeoJI.178.1132L - Synopsis of supervised and unsupervised pattern classification techniques applied to volcanic tremor data at Mt Etna, Italy: for the unsupervised methods, used Cluster Analysis (CA) and SOM
</p>
<p>Clustering of time series data—a survey by T. Warren Liao: 
Clustering methods can be divided into five types, of which partitioning and model-based methods seem the most relevant
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

### Input Data Format

Each column row is a time point, with 12 features corresponding to its Channel 1 and Channel 2 frequency band readings. The Python script to access and compile data into this format can be found as 'getData.py'. 


### Classification Possibilities
<p> Experiment in Matlab with: </p>
<ol>
<li><em>Self-organizing map - unsupervised neural network to find clusters within dataset</em></li>

<li><em>Competitive learning/layers</em></li>
<li><em>k-means clustering</em></li>
</ol>
