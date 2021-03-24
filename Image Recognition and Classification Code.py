
  IMAGE RECOGNITION and CLASSIFICATION MODEL
        
# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib qt

# Get the dataset

from sklearn.datasets import fetch_openml
dataset = fetch_openml('mnist_784')

# MNIST-- Mixed National Institute of Standards and Technology

# It has 70,000 images of handwritten images of 28*28 and encoding HSV.
# its means it is 3d with which ML cannot deal. So, we need to flatten.
# So, the dimension is now 70000 * 784
# X = HSV intensities of all pixels and y = 5 

X = dataset.data
y = dataset.target

y = y.astype('int32')

# Discovery & Visualization

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)

# Since the data is flattened. In order for us to visualize it
# we must again reshape it back to 28 * 28 pixel format.

plt.imshow(some_digit_image)
plt.show()

plt.imshow(some_digit_image, "binary")
plt.axis('off')
plt.show()

# Loop to visualize mutiple images simultaneously

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    im = X[i]
    im = im.reshape(28, 28)
    plt.imshow(im, "binary")
    plt.xlabel("label : {}".format(y[i]))
plt.show()

# Data preprocessing

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Select and train a ML algorithm

################# Logistic Regression #################

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

log_reg.score(X_train, y_train)
log_reg.score(X_test, y_test)

y_pred_log = log_reg.predict(X_test)

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    im = X_test[i]
    im = im.reshape(28, 28)
    plt.imshow(im, "binary")
    plt.xlabel("Actual label : {}\nPredicted label : {}".format(y_test[i], y_pred_log[i]))
plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix
cm_log = confusion_matrix(y_test, y_pred_log)

from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(y_test, y_pred_log, average = "micro")
recall_score(y_test, y_pred_log, average = "micro")
f1_score(y_test, y_pred_log, average = "micro")


####################### Decision Tree ####################

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier()
dtf.fit(X_train, y_train)

dtf.score(X_train, y_train)
dtf.score(X_test, y_test)

y_pred_dtf = dtf.predict(X_test)

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    im = X_test[i]
    im = im.reshape(28, 28)
    plt.imshow(im, "binary")
    plt.xlabel("Actual label : {}\nPredicted label : {}".format(y_test[i], y_pred_dtf[i]))
plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix
cm_dtf = confusion_matrix(y_test, y_pred_dtf)

from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(y_test, y_pred_dtf, average = "micro")
recall_score(y_test, y_pred_dtf, average = "micro")
f1_score(y_test, y_pred_dtf, average = "micro")



