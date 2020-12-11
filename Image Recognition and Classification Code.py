
  IMAGE RECOGNITION and CLASSIFICATION MODEL
        
# Importing the basic libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Get the Data

from sklearn.datasets import fetch_openml
dataset = fetch_openml('mnist_784')

# MNIST-- Mixed National Institute of Standards and Technology

# It has 70,000 images of handwritten images of 28*28 and encoding HSV.
# its means it is 3d with which ML cannot deal. So, we need to flatten.
# So, the dimension is now 784*70000
# X = HSV intensities of all pixels and y = 5 

X = dataset.data
y = dataset.target
y = y.astype(np.int32)

# Since the data is flattened. In order for us to visualize it
# we must again reshape it back to 28 * 28 pixel format.

some_digit = X[69696]
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image)
plt.show()

# sklearn library has fit() predict() score() function
# which makes it very easy to test and apply algo 

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(max_depth = 3)
dtf.fit(X,y)
print(dtf.score(X,y))       # 44.23 %

# its time to get some predictions

y_pred= dtf.predict(X)

# y = vector of observations
# y_pred = vector of predictions

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)  

# 11111 : 3    3936 : 6    7654 : 2    42042 : 4    7777 : 8

y_pred_test = dtf.predict(X[[11111, 3936, 7654, 42042, 7777], 0:784])
print(y_pred_test)              # 3 8 6 1 1

# it could only predict 1 of 5 right as the accuracy is quite low.

# Now, fine tuning.

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(max_depth = 13)
dtf.fit(X,y)
print(dtf.score(X,y))       # 96.34 %


# So, 5 basic steps are complete. 
# We did not follow any optimization techniques 
# which we need to follow while working on an actual project.

