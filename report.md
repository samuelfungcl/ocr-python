# OCR assignment report
## Feature Extraction (Max 200 Words)
The images were first converted to feature vectors using the provided image_to_vectors_ function. 

From there, the feature vectors were reduced to 20 features using get_pca_axes. Initially, I wanted to reduce it to 40 features. However, through my various testing with different values, a reduction to 20 features was the most optimal as it provided higher percentages whilst getting lower runtimes. This in theory, would be better as regardless of reduction of 40 or 20 features, the final 10 features obtained through feature selection will still be the same. Hence, reduction to 20 features was used.

The best feature obtained was paired with other features to calculate their multidivergences, and the pair with the highest multidivergences was then paired with other features until all 10 features were found. I initially obtained the best feature using divergence, but for some reason it provided me with lower percentages. I resulted to using brute force and testing the different features one by one until I got one with a considerable increase in percentage which had the value of 3. Hence, brute force was used instead of divergence.
## Classifier (Max 200 Words)
Both nearest-neighbour and k-nearest neighbour were implemented for the classifier.

For the distance measures, I had previously tested both euclidean distance and cosine distance,and through a couple of experiments it could be seen that cosine distance had a far superior percentage improvement compared to euclidean distance. Hence, cosine distance was used.

The general trend observed was that as the value of k increased, the percentages of the clean data would decrease whilst the percentages of the noisy data would increase. At the end, k-nearest neighbour was chosen with the k value of 11 to ensure that there would not be any errors in labelling which helped improve percentages obtained significantly as well. 

## Error Correction (Max 200 Words)
An attempt at error correction was made but it ended up producing worse/no difference in results and gave a much higher long time.

The individual characters were joined together by checking the gap between the bounding boxes of each character. If the gap between two characters was more than a certain amount (6), the second character would be classified as another word. 

Once the words were seperated, the characters from the test pages were to be compared by the words gotten from the word list to be corrected. The output would then be the corrected words.
## Performance
The percentage errors (to 1 decimal place) for the development data are
as follows:
- Page 1: [95.9%]
- Page 2: [95.5%]
- Page 3: [84.0%]
- Page 4: [61.6%]
- Page 5: [42.7%]
- Page 6: [36.7%]
## Other information (Optional, Max 100 words)
[Optional: highlight any significant aspects of your system that are
NOT covered in the sections above]