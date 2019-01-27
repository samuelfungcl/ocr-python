# Data Driven Computing - Python
# Optical Character Recogniser

## Description
The project was to demonstrate machine learning algorithms and techniques to implement an optical character recogniser by using feature reduction and feature selection algorithms to enhance efficiency of the system

## Task
The system was given some training data and was required to compute the features that were relevant in identifying the characters in the pdf pages and to store these features that were then used to test on some sample data/test pages. The system was implemented in python using a K-Nearest Neighbour classifier and Principal Component Analysis for feature reduction. From there, a forward sequential feature selection was used to get the best features by comparing their divergences and multidivergences and an error correction algorithm was implemented as well. 

## To run
Training mode: python train.py
Evaluation mode: python evalute.py dev

## Test Results
You could add the argument "dev" to the application when running it. This will re setup the database with dummy data.
- Page 1: [95.9%]
- Page 2: [95.5%]
- Page 3: [84.0%]
- Page 4: [61.6%]
- Page 5: [42.7%]
- Page 6: [36.7%]

## Marks Obtained
I received a first class mark (78%) for this assignment
