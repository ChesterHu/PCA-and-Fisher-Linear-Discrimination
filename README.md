# PCA and Fisher Linear Discrimination for human faces
Various methods for image classification and pattern recognition
'''
data type: uint8 256*256
'''

Part 1: PCA
1. Using PCA find eigen-faces of training set(150 face). Then use eigen-faces to reconstruct human faces in test set(27 faces).
2. Using PCA find eigen-landmarks of training set, then reconstruct landmarks for test set.
3. Combine two methods above: first find eigen-landmarks of training faces, then warp all faces to mean position(the geometric feature of    the mean-face), find eigen-faces of the warpped faces. Using the eigen-landmarks and eigen-faces to reconstruct the testing faces.

Part 2: FLD
1. Find Fisher faces for the training set for gender classification, and use the projection w to discriminate gender of testing faces.
2. First warp all faces to the mean position, find Fisher faces for training faces and training landmarks. Then construct a 2-D graph to      discriminate gender on testing faces.
