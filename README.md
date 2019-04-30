# Low-Light-Object-Classification
Course project for Computer Vision (CSE/ECE 344/544) course at IIIT-D

# About
Automated object classification has a variety of applications in real world. Deep Neural Networks perform well in the task of object classification but their performance in low-light conditions is not thoroughly tested. 
We aim to improve existing low-light classification technique using a preprocessing pipeline and a fine-tuned deep CNN model.

# Use Case
A large number of accidents and injuries occur because of the reduced visibility in low-light, particularly during night
time. Automated object classification can aid drivers to make timely decisions.
Deep sea exploration and wildlife animal classification are difficult tasks when it is not feasible to obtain good quality
images. We intend to improve classification in such scenarios through our work.

# Literature review

# Database details

## ExDark
- Color images of varying size
- We have converted all the images to grayscale and a fixed reduced size
- Add more

## Face Dataset
- Grayscale images of fixed size
- We have converted all the images to a fixed reduced size
- Add more

# Tasks

## Completed
0. Enhance low light noisy images using contrast enhancement techniques presented by [2] and [3].  
    - Comparing and contrasting various contrast enhancement methods.   
    - Performed on various low light noisy images and then improvement is assessed for all methods.  
1. Analysis of effect of low light images on classical classification methods (using non-neural classifiers).
- Training the models with low light images and processed bright images (using all enhancement methods) to identify the performance difference. This is performed on ExDARK and Face Datasets.  
    Classification models:
    - Naive Bayes Classifier
    - Random Forest Classifier
    - K Nearest Neighbours Classfier  
- Pixel intensity values are used as features for classification.   
- We select the best method for processing low light images by using classification accuracy as a metric.
2. Add more

## To be done
0. Repeat the classfication using colour images and using a combination of global and local feature vectors - bag of visual words approach (BoVW).
1. Using the selected enhancement method we will perform classification using variants of convolutional neural networks. The variants will be selected as per their performance on the COCO dataset [4].
2. We optimise our pre-processing pipeline and fine tune the selected variants of neural networks and other classification models.
2. Add more 

# Reference

1. [Loh, Y.P. and Chan, C.S., 2019. Getting to know low-light images with the Exclusively Dark dataset. Computer Vision and Image Understanding, 178, pp.30-42.](https://arxiv.org/abs/1805.11227)

2. [Abdullah-Al-Wadud, M., Kabir, M.H., Dewan, M.A.A. and Chae, O., 2007. A dynamic histogram equalization for image contrast enhancement. IEEE Transactions on Consumer Electronics, 53(2), pp.593-600.](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4266947)

3. [Ying, Z., Li, G., Ren, Y., Wang, R. and Wang, W., 2017, August. A new image contrast enhancement algorithm using exposure fusion framework. In International Conference on Computer Analysis of Images and Patterns (pp. 36-46). Springer, Cham.](https://link.springer.com/chapter/10.1007/978-3-319-64698-5_4)

4. [COCO: Common Objects in Context](http://cocodataset.org/#home)