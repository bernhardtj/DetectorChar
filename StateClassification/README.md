# Classification of Interferometer States



### Ideas

1. What is the seismic state? (e.g. EQ, truck, windy, broken sensor)
1. Are the optical levers going bad? (e.g. look at the noise at 1 Hz vs. the noise at 50 Hz)

![ML Cheat Sheet](microsoft-machine-learning-algorithm-cheat-sheet-v6.png?raw=true "Title")

### Input Data Format

Each column is one time series for a specific frequency band. The points in the time series are
vectors to incorporate data from multiple channels. Python script to get data and organize into this format?

|  Time  | 0.03 - 0.1 Hz |  0.1 - 0.3 Hz | 0.3 - 1 Hz
|--------|:-------------:|:-------------:|:----------:
|    1   |   {ch1, ch2}  |   {ch1, ch2}  | {ch1, ch2}
|    2   |   {ch1, ch2}  |   {ch1, ch2}  | {ch1, ch2}
|    3   |   {ch1, ch2}  |   {ch1, ch2}  | {Ch1, ch2}
|    4   |   {ch1, ch2}  |   {ch1, ch2}  | {ch1, ch2}
|    5   |   {ch1, ch2}  |   {ch1, ch2}  | {ch1, ch2}

### Classification Possibilities
<p> Experiment in Matlab with: </p>
<ol>
<li><em>Self-organizing map</em></li>
<li><em>Competitive learning/layers</em></li>
<li><em>k-means clustering</em></li>
</ol>
