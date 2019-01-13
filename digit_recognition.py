{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "#import the files\n",
    "training = pd.read_csv('mnist_train.csv')\n",
    "test = pd.read_csv('mnist_test.csv')\n",
    "\n",
    "\n",
    "X_train = training.values[:,1:]\n",
    "y_train = training.values[:,0]\n",
    "m = y_train.shape[0]\n",
    "\n",
    "X_test = test.values[:,1:]\n",
    "y_test = test.values[:,0]\n",
    "\n",
    "from sklearn.svm import SVC\n",
    "classifier = SVC(kernel = 'linear')\n",
    "\n",
    "# Learning Phase\n",
    "classifier.fit(X_train, y_train)\n",
    "\n",
    "# Predict Test Set\n",
    "predicted = classifier.predict(X_test)\n",
    "\n",
    "#evaluating the result using confusion matrix\n",
    "#This may take a while\n",
    "from sklearn.metrics import confusion_matrix\n",
    "cm = confusion_matrix(y_test,predicted)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
