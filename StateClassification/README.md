# Classification of Interferometer States



### Ideas

1. What is the seismic state? (e.g. EQ, truck, windy, broken sensor)
1. Are the optical levers going bad? (e.g. look at the noise at 1 Hz vs. the noise at 50 Hz)

![ML Cheat Sheet](microsoft-machine-learning-algorithm-cheat-sheet-v6.png?raw=true "Title")

### Input Data Format

Each column row is a time point, with 12 features corresponding to its Channel 1 and Channel 2 frequency band readings. The Python script to access and compile data into this format can be found as 'getData.py'. 


### Classification Possibilities
<p> Experiment in Matlab with: </p>
<ol>
<li><em>Self-organizing map - unsupervised neural network to find clusters within dataset</em></li>

<li><em>Competitive learning/layers</em></li>
<li><em>k-means clustering</em></li>
</ol>
