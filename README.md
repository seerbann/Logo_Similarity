# Logo Similarity
## Program created with the objective to document how different feature descriptors, the Structural Similarity Index Measure and different CNN models behave on clustering a set of logos crawled from the web. Also not using any clustering algorithm like DBSCAN or k-means.

### <u> Feature descriptors</u> used:
- **ORB (Oriented FAST and Rotated Brief)** [ORB - OpenCV docs](https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html)  <br/>
- **SIFT (Scale Invariant Feature Transform)**  [SIFT-Wikipedia](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform)<br/>
- **pHASH (Perceptual Hashing)**  [Perceptual Hashing-Wikipedia](https://en.wikipedia.org/wiki/Perceptual_hashing)<br/>
        A perceptual hash is a fingerprint of a multimedia file derived from various features from its content. Unlike cryptographic hash functions which rely on the avalanche effect of small changes in input leading to drastic changes in the output, perceptual hashes are "close" to one another if the features are similar. <br/>
- **pHash and ORB combined** <br/>
        For 2 images to be considered similar they need to have similar ORB scores and also simillar pHash scores. This is a good combination because ORB gets the local details and pHash keeps track of the general aspect of the logo.
###  <u> Structural Similarity Index (SSIM) </u> 
[Structure Similarity Index - SciKit docs](https://scikit-image.org/docs/0.25.x/auto_examples/transform/plot_ssim.html) <br/>
Method that compares 2 pictures at a global level, meaning that it analyses contrast, structure and lightning. It is able to detected small differences in color or contrast. 
### <u> CNN pre-trained models</u> used:
- **ResNet50**
- **EfficientNet B0**
