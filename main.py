"""
Two steps to define similarity

1. One is compare the features característics and base on that
define the similarity as the euclidian distance between the features vector, some
weights could be appply to the importance of the features of the vector if neccesary.
This first step will have a threshold value 

2. Applais the same principle to define similarity but the extraction of the audio features
are extracted by a siamnese neuronal network, here the distance is evaluated in the latent
space of neuronal networks with identical architecture, I have to define if the data will
be the spectograms of the audio in a CNN arch or the featuresextracted and analyzed with a 
DNN model.

The first step will segment the sounds that are similar (in principal with low compu time)
and then the refine process of similary is define by the nueronal network model.
"""

