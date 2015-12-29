# CSinglePerceptron

Michael Krumdick

HOW TO RUN:
$ make
$ ./neural iris.txt
		or
$ ./neural cancer.txt

WHAT WILL HAPPEN:
 The program will display its final set of classificaitons and its average success rate
 over all of its tests. It may take a few seconds. This average value should be about 93%
 for iris.txt and 90% for cancer.txt.

WHAT ARE THE TWO DATA SETS:

iris.txt-
	The iris flower data set. This data set, commonly used in machine learning, contains
	the measurements of various parts of a flower and its species. The program identifies
	which type of flower matches the data.

cancer.txt-
	This dataset contains information about breast cancer tumors. It predicts whether or
	not they are malignant. Data from:
	https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

DIRECTIONS FOR ADDING ANOTHER DATASET:
	If you wished to use this program for another data set, on the first line of
	file write how many values that each point has, along with the number of possible
	classifications seperated by a space. The data points need to be real numbers. The
	classifiers can be strings or numbers.
