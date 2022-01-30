# Import necessary modules
from sklearn import datasets
import matplotlib.pyplot as plt
 
# Load the digits dataset: digits
digits = datasets.load_digits()
 
# Print the keys and DESCR of the dataset
print(digits.keys())
#dict_keys(['data', 'target', 'target_names', 'images', 'DESCR'])
 
print(digits.DESCR)
/*
Optical Recognition of Handwritten Digits Data Set
===================================================
 
Notes
-----
Data Set Characteristics:
    :Number of Instances: 5620
    :Number of Attributes: 64
    :Attribute Information: 8x8 image of integer pixels in the range 0..16.
    :Missing Attribute Values: None
    :Creator: E. Alpaydin (alpaydin '@' boun.edu.tr)
    :Date: July; 1998
...
*/
 
# Print the shape of the images and data keys
print(digits.images.shape)
(1797, 8, 8)
 
print(digits.data.shape)
(1797, 64)
 
# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()