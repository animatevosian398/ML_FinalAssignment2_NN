#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import warnings

warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from skimage import io, color, transform, feature
from my_measures import BinaryClassificationPerformance  
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# ### IMPORTANT!!! Make sure you are using BinaryClassificationPerformance v1.03
# 

# In[2]:


help(BinaryClassificationPerformance)


# ### file paths and names

# In[3]:


ci_path = 'plane_data/cropped_images/' # file path for cropped images for training
l_file = 'plane_data/plane_labels.csv' # file path and file name for csv with labels


# # Function for feature building and extraction on photographs¶
# 
# scikit-image documentation on methods used for feature extraction:  
# 
# * http://scikit-image.org/docs/dev/api/skimage.color.html#rgb2gray  
# * http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize  
# * http://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.canny

# In[4]:


# # in downscaling the image, what do you want the new dimensions to be?
# # the original dimensions of cropped images: (60, 140), which is 8,400 pixels
# dims = (15, 35) # x of the original size, 525 pixels


# In[5]:


# in downscaling the image, what do you want the new dimensions to be?
# the original dimensions of cropped images: (60, 140), which is 8,400 pixels
dims = (60, 140) #  original size


# In[6]:


##########gaussian AND sobel##########
from skimage import io, transform, filters

def image_manipulation(imname, imgs_path, imview=False):
    warnings.filterwarnings('ignore')
    imname = imgs_path + imname + '.png'
    img_raw = io.imread(imname, as_gray=True)
    downscaled = transform.resize(img_raw, (dims[0], dims[1]))
    # downscale image
    smoothed = filters.gaussian(downscaled)  # gaussian smoothed sigma=1
    final_image = filters.sobel(smoothed) #SOBEL
#     final_image = filters.farid(downscaled);
#     final_image = feature.canny(downscaled) # edge filter image with Canny algorithm
    if imview==True:
        io.imshow(final_image)
    warnings.filterwarnings('always')
    return final_image

# test the function, look at input/output
test_image = image_manipulation('2017-08-25T23+24+13_390Z', ci_path, True)
print('downscaled image shape: ')
print(test_image.shape)
print('image representation (first row of pixels): ')
print(test_image[0])
print('\n')
print('example of transformation: ')


# In[7]:


# ##########Just gaussian##########
# from skimage import io, transform, filters

# def image_manipulation(imname, imgs_path, imview=False):
#     warnings.filterwarnings('ignore')
#     imname = imgs_path + imname + '.png'
#     img_raw = io.imread(imname, as_gray=True)
#     downscaled = transform.resize(img_raw, (dims[0], dims[1])) # downscale image
#     final_image = filters.gaussian(downscaled); #
# #     final_image = filters.farid(downscaled);
# #     final_image = feature.canny(downscaled) # edge filter image with Canny algorithm
#     if imview==True:
#         io.imshow(final_image)
#     warnings.filterwarnings('always')
#     return final_image

# # test the function, look at input/output
# test_image = image_manipulation('2017-08-25T23+24+13_390Z', ci_path, True)
# print('downscaled image shape: ')
# print(test_image.shape)
# print('image representation (first row of pixels): ')
# print(test_image[0])
# print('\n')
# print('example of transformation: ')


# for comparison, look at original image:

# In[8]:


# this_imname = ci_path + '2017-08-25T23+24+13_390Z.png'
# io.imshow(io.imread(this_imname))


# # function to process raw images, resulting in training and test datasets

# In[9]:


# function that takes raw images and completes all preprocessing required before model fits
def process_raw_data(labels_fn, images_fp, my_random_seed, imview=False, test=False):
    plane_data = pd.read_csv(labels_fn) # read in photograph labels
    print("First few lines of image labels: ")
    print(plane_data.head())
    print("Size of image label dataFrame: ")
    print(plane_data.shape)
        
    # construct lists for features, labels, and a crosswalk reference to image names
    features_list = []
    if (not test):
        y_list = []
    imnames_list = []

    for index, row in plane_data.iterrows():
        features_list.append(image_manipulation(row['img_name'], images_fp))
        if (not test):
            y_list.append(row['plane'])
        imnames_list.append(row['img_name'])
    
    # convert the lists to ndarrays
    features = np.asarray(features_list)
    if (not test):
        Y = np.asarray(y_list)
    imgs = np.asarray(imnames_list)
    print('Shape of original feature representation: ')
    print(features.shape)

    # flatten the images ndarray to one row per image
    features_flat = features.reshape((features.shape[0], -1))

    print('Shape of flat feature representation: ')
    print(features_flat.shape)

    if (not test):
        print('Shape of Y: ')
        print(Y.shape)

        print('Number of images with planes: ')
        print(Y.sum())
    
        # create train and test sets
        data_train, data_test, y_train, y_test, imgs_train, imgs_test = train_test_split(features_flat, 
            Y, imgs, test_size = 0.25, random_state = my_random_seed)

        print('Shape of training set: ')
        print(y_train.shape)
        print('Number of training images that contain an airplane: ')
        print(y_train.sum())

        print('Shape of test set: ')
        print(y_test.shape)
        print('Number of test images that contain an airplane: ')
        print(y_test.sum())
    
    if (test):
        X_submission_test = features_flat
        print("Shape of X_test for submission:")
        print(X_submission_test.shape)
        print('SUCCESS!')
        return(X_submission_test, plane_data)
    else: 
        print("Shape of data_train and data_test:")
        print(data_train.shape)
        print(data_test.shape)
        print("Shape of y_train and y_test:")
        print(y_train.shape)
        print(y_test.shape)
        print("Shape of imgs_train and imgs_test:")
        print(imgs_train.shape)
        print(imgs_test.shape)
        print('SUCCESS!')
        return(data_train, data_test, y_train, y_test, imgs_train, imgs_test)


# In[10]:


data_train, data_test, y_train, y_test, imgs_train, imgs_test = process_raw_data(l_file, ci_path, 
    my_random_seed=1626, imview=False, test=False)


# # train Perceptron

# In[11]:


# # MODEL: Perceptron
# from sklearn import linear_model
# prc = linear_model.SGDClassifier(loss='perceptron')
# prc.fit(data_train, y_train)

# prc_performance = BinaryClassificationPerformance(prc.predict(data_train), y_train, 'prc')
# prc_performance.compute_measures()
# prc_performance.performance_measures['set'] = 'train'
# print('TRAINING SET: ')
# print(prc_performance.performance_measures)

# prc_performance_test = BinaryClassificationPerformance(prc.predict(data_test), y_test, 'prc')
# prc_performance_test.compute_measures()
# prc_performance_test.performance_measures['set'] = 'test'
# print('TEST SET: ')
# print(prc_performance_test.performance_measures)

# prc_performance_test.img_indices()
# prc_img_indices_to_view = prc_performance_test.image_indices


# In[12]:


def performance_examples(typ, measures):
    iiv = ''
    if typ == 'FP':
        iiv = typ + '_indices'
    elif typ == 'TP':
        iiv = typ + '_indices'
    elif typ == 'FN':
        iiv = typ + '_indices'
    else:
        raise ValueError('input must be "TP", "FP", or "FN"')
    for img in measures[iiv]:
        warnings.filterwarnings('ignore')    
        plt.figure()
        lookat = ci_path + imgs_test[img] + '.png' # location of original image
        io.imshow(lookat) # show original image
        plt.figure()
        io.imshow(data_test[img].reshape(dims[0], dims[1])) # show manipulation for feature representation
        warnings.filterwarnings('always')


# # look at examples of Perceptron classifications

# ## true positives

# In[13]:


# performance_examples('TP', prc_img_indices_to_view)


# ## false positives

# In[14]:


# performance_examples('FP', prc_img_indices_to_view)


# ## false negatives

# In[15]:


# performance_examples('FN', prc_img_indices_to_view)


# # train Multilayer Perceptron, a.k.a. neural network
# 
# 

# In[16]:


# Default hidden layer size = 100,
#MODEL: Multi-layer Perceptron aka neural network
from sklearn import neural_network
nn = neural_network.MLPClassifier(max_iter=1000)
print(nn)
nn.fit(data_train, y_train)

nn_performance = BinaryClassificationPerformance(nn.predict(data_train), y_train, 'nn')
nn_performance.compute_measures()
nn_performance.performance_measures['set'] = 'train'
print('TRAINING SET: ')
print(nn_performance.performance_measures)

nn_performance_test = BinaryClassificationPerformance(nn.predict(data_test), y_test, 'nn_test')
nn_performance_test.compute_measures()
nn_performance_test.performance_measures['set'] = 'test'
print('TEST SET: ')
print(nn_performance_test.performance_measures)

nn_performance_test.img_indices()
nn_img_indices_to_view = nn_performance_test.image_indices


# In[17]:


# 50, 50
#MODEL: Multi-layer Perceptron aka neural network
from sklearn import neural_network
nn_50_50 = neural_network.MLPClassifier(hidden_layer_sizes=(50,50),max_iter=1000)
print(nn_50_50)
nn_50_50.fit(data_train, y_train)

nn_performance_50_50 = BinaryClassificationPerformance(nn_50_50.predict(data_train), y_train, 'nn_50_50')
nn_performance_50_50.compute_measures()
nn_performance_50_50.performance_measures['set'] = 'train'
print('TRAINING SET: ')
print(nn_performance_50_50.performance_measures)

nn_performance_test_50_50 = BinaryClassificationPerformance(nn_50_50.predict(data_test), y_test, 'nn_test_50_50')
nn_performance_test_50_50.compute_measures()
nn_performance_test_50_50.performance_measures['set'] = 'test'
print('TEST SET: ')
print(nn_performance_test_50_50.performance_measures)

nn_performance_test_50_50.img_indices()
nn_img_indices_to_view_50_50= nn_performance_test_50_50.image_indices


# In[18]:


#(128, 64, 32)
#MODEL: Multi-layer Perceptron aka neural network
from sklearn import neural_network
nn_128_64_32 = neural_network.MLPClassifier(hidden_layer_sizes=(128, 64, 32),max_iter=1000)
print(nn_128_64_32)
nn_128_64_32.fit(data_train, y_train)

nn_performance_128_64_32 = BinaryClassificationPerformance(nn_128_64_32.predict(data_train), y_train, 'nn_128_64_32')
nn_performance_128_64_32.compute_measures()
nn_performance_128_64_32.performance_measures['set'] = 'train'
print('TRAINING SET: ')
print(nn_performance_128_64_32.performance_measures)

nn_performance_test_128_64_32 = BinaryClassificationPerformance(nn_128_64_32.predict(data_test), y_test, 'nn_test_128_64_32')
nn_performance_test_128_64_32.compute_measures()
nn_performance_test_128_64_32.performance_measures['set'] = 'test'
print('TEST SET: ')
print(nn_performance_test_128_64_32.performance_measures)

nn_performance_test_128_64_32.img_indices()
nn_img_indices_to_view_128_64_32= nn_performance_test_128_64_32.image_indices


# In[19]:


# 50, 25
#MODEL: Multi-layer Perceptron aka neural network
from sklearn import neural_network
nn_50_25 = neural_network.MLPClassifier(hidden_layer_sizes=(50,25),max_iter=1000)
print(nn_50_25)
nn_50_25.fit(data_train, y_train)

nn_performance_50_25 = BinaryClassificationPerformance(nn_50_25.predict(data_train), y_train, 'nn_50_25')
nn_performance_50_25.compute_measures()
nn_performance_50_25.performance_measures['set'] = 'train'
print('TRAINING SET: ')
print(nn_performance_50_25.performance_measures)

nn_performance_test_50_25 = BinaryClassificationPerformance(nn_50_25.predict(data_test), y_test, 'nn_test_50_25')
nn_performance_test_50_25.compute_measures()
nn_performance_test_50_25.performance_measures['set'] = 'test'
print('TEST SET: ')
print(nn_performance_test_50_25.performance_measures)

nn_performance_test_50_25.img_indices()
nn_img_indices_to_view_50_25= nn_performance_test_50_25.image_indices


# In[20]:


# 100, 50, 25
#MODEL: Multi-layer Perceptron aka neural network
from sklearn import neural_network
nn_100_50_25 = neural_network.MLPClassifier(hidden_layer_sizes=(100,50,25),max_iter=1000)
print(nn_100_50_25)
nn_100_50_25.fit(data_train, y_train)

nn_performance_100_50_25 = BinaryClassificationPerformance(nn_100_50_25.predict(data_train), y_train, 'nn_100_50_25')
nn_performance_100_50_25.compute_measures()
nn_performance_100_50_25.performance_measures['set'] = 'train'
print('TRAINING SET: ')
print(nn_performance_100_50_25.performance_measures)

nn_performance_test_100_50_25 = BinaryClassificationPerformance(nn_100_50_25.predict(data_test), y_test, 'nn_test_100_50_25')
nn_performance_test_100_50_25.compute_measures()
nn_performance_test_100_50_25.performance_measures['set'] = 'test'
print('TEST SET: ')
print(nn_performance_test_100_50_25.performance_measures)

nn_performance_test_100_50_25.img_indices()
nn_img_indices_to_view_100_50_25 = nn_performance_test_100_50_25.image_indices


# In[21]:


# 50, 30, 10
#MODEL: Multi-layer Perceptron aka neural network
from sklearn import neural_network
nn_50_30_10 = neural_network.MLPClassifier(hidden_layer_sizes=(50,30,10),max_iter=1000)
print(nn_50_30_10)
nn_50_30_10.fit(data_train, y_train)

nn_performance_50_30_10 = BinaryClassificationPerformance(nn_50_30_10.predict(data_train), y_train, 'nn_50_30_10')
nn_performance_50_30_10.compute_measures()
nn_performance_50_30_10.performance_measures['set'] = 'train'
print('TRAINING SET: ')
print(nn_performance_50_30_10.performance_measures)

nn_performance_test_50_30_10 = BinaryClassificationPerformance(nn_50_30_10.predict(data_test), y_test, 'nn_test_50_30_10')
nn_performance_test_50_30_10.compute_measures()
nn_performance_test_50_30_10.performance_measures['set'] = 'test'
print('TEST SET: ')
print(nn_performance_test_50_30_10.performance_measures)

nn_performance_test_50_30_10.img_indices()
nn_img_indices_to_view_50_30_10 = nn_performance_test_50_30_10.image_indices


# In[22]:


# 80, 40, 20
#MODEL: Multi-layer Perceptron aka neural network
from sklearn import neural_network
nn_80_40_20 = neural_network.MLPClassifier(hidden_layer_sizes=(80,40,20),max_iter=1000)
print(nn_80_40_20)
nn_80_40_20.fit(data_train, y_train)

nn_performance_80_40_20 = BinaryClassificationPerformance(nn_80_40_20.predict(data_train), y_train, 'nn_80_40_20')
nn_performance_80_40_20.compute_measures()
nn_performance_80_40_20.performance_measures['set'] = 'train'
print('TRAINING SET: ')
print(nn_performance_80_40_20.performance_measures)

nn_performance_test_80_40_20 = BinaryClassificationPerformance(nn_80_40_20.predict(data_test), y_test, 'nn_test_80_40_20')
nn_performance_test_80_40_20.compute_measures()
nn_performance_test_80_40_20.performance_measures['set'] = 'test'
print('TEST SET: ')
print(nn_performance_test_80_40_20.performance_measures)

nn_performance_test_80_40_20.img_indices()
nn_img_indices_to_view_80_40_20 = nn_performance_test_80_40_20.image_indices


# In[23]:


# 100, 50, 25, 12
#MODEL: Multi-layer Perceptron aka neural network
from sklearn import neural_network
nn_100_50_25_12 = neural_network.MLPClassifier(hidden_layer_sizes=(100,50,25,12),max_iter=1000)
print(nn_100_50_25_12)
nn_100_50_25_12.fit(data_train, y_train)

nn_performance_100_50_25_12 = BinaryClassificationPerformance(nn_100_50_25_12.predict(data_train), y_train, 'nn_100_50_25_12')
nn_performance_100_50_25_12.compute_measures()
nn_performance_100_50_25_12.performance_measures['set'] = 'train'
print('TRAINING SET: ')
print(nn_performance_100_50_25_12.performance_measures)

nn_performance_test_100_50_25_12 = BinaryClassificationPerformance(nn_100_50_25_12.predict(data_test), y_test, 'nn_test_100_50_25_12')
nn_performance_test_100_50_25_12.compute_measures()
nn_performance_test_100_50_25_12.performance_measures['set'] = 'test'
print('TEST SET: ')
print(nn_performance_test_100_50_25_12.performance_measures)

nn_performance_test_100_50_25_12.img_indices()
nn_img_indices_to_view_100_50_25_12 = nn_performance_test_100_50_25_12.image_indices


# In[24]:


# 200, 100, 50, 25, 12
#MODEL: Multi-layer Perceptron aka neural network
from sklearn import neural_network
nn_200_100_50_25_12 = neural_network.MLPClassifier(hidden_layer_sizes=(200,100,50,25,12),max_iter=1000)
print(nn_200_100_50_25_12)
nn_200_100_50_25_12.fit(data_train, y_train)

nn_performance_200_100_50_25_12 = BinaryClassificationPerformance(nn_200_100_50_25_12.predict(data_train), y_train, 'nn_200_100_50_25_12')
nn_performance_200_100_50_25_12.compute_measures()
nn_performance_200_100_50_25_12.performance_measures['set'] = 'train'
print('TRAINING SET: ')
print(nn_performance_200_100_50_25_12.performance_measures)

nn_performance_test_200_100_50_25_12 = BinaryClassificationPerformance(nn_200_100_50_25_12.predict(data_test), y_test, 'nn_test_200_100_50_25_12')
nn_performance_test_200_100_50_25_12.compute_measures()
nn_performance_test_200_100_50_25_12.performance_measures['set'] = 'test'
print('TEST SET: ')
print(nn_performance_test_200_100_50_25_12.performance_measures)

nn_performance_test_200_100_50_25_12.img_indices()
nn_img_indices_to_view_200_100_50_25_12 = nn_performance_test_200_100_50_25_12.image_indices


# In[25]:


# 400, 200, 100, 50, 25, 12
#MODEL: Multi-layer Perceptron aka neural network
from sklearn import neural_network
nn_400_200_100_50_25_12 = neural_network.MLPClassifier(hidden_layer_sizes=(400,200,100,50,25,12),max_iter=1000)
print(nn_400_200_100_50_25_12)
nn_400_200_100_50_25_12.fit(data_train, y_train)

nn_performance_400_200_100_50_25_12 = BinaryClassificationPerformance(nn_400_200_100_50_25_12.predict(data_train), y_train, 'nn_400_200_100_50_25_12')
nn_performance_400_200_100_50_25_12.compute_measures()
nn_performance_400_200_100_50_25_12.performance_measures['set'] = 'train'
print('TRAINING SET: ')
print(nn_performance_400_200_100_50_25_12.performance_measures)

nn_performance_test_400_200_100_50_25_12 = BinaryClassificationPerformance(nn_400_200_100_50_25_12.predict(data_test), y_test, 'nn_test_400_200_100_50_25_12')
nn_performance_test_400_200_100_50_25_12.compute_measures()
nn_performance_test_400_200_100_50_25_12.performance_measures['set'] = 'test'
print('TEST SET: ')
print(nn_performance_test_400_200_100_50_25_12.performance_measures)

nn_performance_test_400_200_100_50_25_12.img_indices()
nn_img_indices_to_view_400_200_100_50_25_12 = nn_performance_test_400_200_100_50_25_12.image_indices


# In[ ]:





# # look at examples of neural network classifications

# ## true positives

# In[26]:


# performance_examples('TP', nn_img_indices_to_view)


# In[27]:


# performance_examples('TP', nn_img_indices_to_view_100_50_25)


# ## false positives

# In[28]:


# performance_examples('FP', nn_img_indices_to_view)


# In[29]:


# performance_examples('FP', nn_img_indices_to_view_100_50_25)


# ## false negatives

# In[30]:


# performance_examples('FN', nn_img_indices_to_view_100_50_25)


# # comparisons

# In[34]:


# list of fits to compare: 
final_fits = []
# final_fits.append(prc_performance.performance_measures)
# final_fits.append(prc_performance_test.performance_measures)

# final_fits.append(nn_performance.performance_measures)
# final_fits.append(nn_performance_test.performance_measures)

# final_fits.append(nn_performance_50_25.performance_measures)
# final_fits.append(nn_performance_test_50_25.performance_measures)

# final_fits.append(nn_performance_50_50.performance_measures)
# final_fits.append(nn_performance_test_50_50.performance_measures)

# final_fits.append(nn_performance_128_64_32.performance_measures)
# final_fits.append(nn_performance_test_128_64_32.performance_measures)

# final_fits.append(nn_performance_50_30_10.performance_measures)
# final_fits.append(nn_performance_test_50_30_10.performance_measures)

# final_fits.append(nn_performance_80_40_20.performance_measures)
# final_fits.append(nn_performance_test_80_40_20.performance_measures)

# final_fits.append(nn_performance_100_50_25.performance_measures)
# final_fits.append(nn_performance_test_100_50_25.performance_measures)

# # final_fits.append(nn_performance_50_10.performance_measures)
# # final_fits.append(nn_performance_test_50_10.performance_measures)

# final_fits.append(nn_performance_100_50_25_12.performance_measures)
# final_fits.append(nn_performance_test_100_50_25_12.performance_measures)

# final_fits.append(nn_performance_200_100_50_25_12.performance_measures)
# final_fits.append(nn_performance_test_200_100_50_25_12.performance_measures)

final_fits.append(nn_performance_400_200_100_50_25_12.performance_measures)
final_fits.append(nn_performance_test_400_200_100_50_25_12.performance_measures)



# In[35]:


plt.figure(figsize=(10,10))

for fit in final_fits:
    if fit['set'] == 'train':
        color = 'co'
    else:
        color = 'ro'
    plt.plot(fit['FP'] / fit['Neg'], 
             fit['TP'] / fit['Pos'], color, markersize=12)
    plt.text(fit['FP'] / fit['Neg'], 
             fit['TP'] / fit['Pos'], fit['desc'] + ': ' + fit['set'], fontsize=16)
plt.axis([0, 1, 0, 1])
plt.title('ROC plot: test set')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.show()


# # SUBMISSION
# 
# ### file paths and names:

# In[ ]:


submission_ci_path = 'test_data_for_grading/test_cropped_images/' # file path for cropped images for training
submission_l_file = 'test_data_for_grading/test_plane_labels.csv' # file path and file name for csv with labels


# In[ ]:


X_test_data, X_test_submission = process_raw_data(submission_l_file, submission_ci_path, my_random_seed=1626, test=True)
print("Number of rows in the submission test set (should be 1,523): ")


# ### IMPORTANT CHECK: make sure that the number of columns in your training data is the same as the number of columns in this test submission!

# In[ ]:


print(data_train.shape)
print(X_test_data.shape)


# Both the training set and submission test set have 525 columns. Success!

# ---
# 
# Choose a *single* model for your submission. In this code, I am choosing the Perceptron model fit, which is in the prc object. But you should choose the model that is performing the best for you!

# In[ ]:


# # concatenate predictions to the id
# X_test_submission["prediction"] = prc.predict(X_test_data)
# # look at the proportion of positive predictions
# print(X_test_submission['prediction'].mean())


# In[ ]:


# concatenate predictions to the id
X_test_submission["prediction"] = nn.predict(X_test_data)
# look at the proportion of positive predictions
print(X_test_submission['prediction'].mean())


# This is the proportion of predictions that have predicted that there is an airplane in the image.

# In[ ]:


print(X_test_submission.shape) # should be (1523, 2)


# In[ ]:


# # export submission file as pdf
# # CHANGE FILE PATH: 
# X_test_submission.to_csv('airplane_submission.csv', index=False)

