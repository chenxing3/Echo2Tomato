# Echo2Tomato
## Determining Tomato Count via Robot Echoes
In this project, I developed a GPT-2 model designed to detect specific frequency sequences within spectrograms. These spectrograms were generated from echo waveforms, which consisted of both transmitted and received signal components. To standardize the signal components, I employed the 'convolve' method, using an ideal chirp as a reference. The standardized chirps were then transformed into spectrograms with time-varying frequency patterns. To identify these frequency patterns, I applied hierarchical cluster analysis using Ward's method.


# Install
