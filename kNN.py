import cv2
import numpy as np
import os

# path to the data set
dir_train_data = os.fsencode("training_data")

# number of digits to train
n = 10
# k in k-Nearest Neighbour (number of neighbours to consider for classification)
k = 5
# total number of samples
total_data = 1000
# number of samples to use as training data (testing data will be calculated accordingly)
no_train_data = 700
# for pre-processing, image transformation
# resultant width of the sample image
data_width = 20
# resultant height of the sample image
data_height = 20

# labels for the training data classes
numbers = np.arange(n)
# counters to keep track of the training and testing data sets
noOfTraining = 0
noOfTesting = 0

# training samples (to be updated from the image inputs
img_train = np.empty([no_train_data*n, data_width*data_height])
img_test = np.empty([(total_data-no_train_data)*n, data_width*data_height])

print("Loading digits...")
# iterate through the directory to get digit samples
for digit_samples in numbers:
    i = 0
    # iterate through the image samples
    for digits in os.listdir(os.fsdecode(dir_train_data) + "/" + str(digit_samples)):
        # get the sample image
        img = cv2.imread(os.fsdecode(dir_train_data) + "/" + str(digit_samples) + "/" + os.fsdecode(digits), 0)
        # resize it
        r_img = cv2.resize(img, (data_width, data_height), interpolation=cv2.INTER_CUBIC)
        # reduce noise (canny edge detection) (test case: 1)
        # c_img = cv2.Canny(r_img, 100, 200)
        # remove the erosion (test case: 2)
        kernel = np.ones((5,5),np.uint8)
        c_img = cv2.erode(r_img, kernel, iterations=1)
        # convert it to array
        if i < no_train_data:
            img_train[noOfTraining] = np.array(c_img).reshape(data_width * data_height)
            noOfTraining += 1
        else:
            img_test[noOfTesting] = np.array(c_img).reshape(data_width * data_height)
            noOfTesting += 1
        i += 1

# convert it into d dimensional array of 400. (where d is number of train data and 400 is size of image)
img_train = img_train.reshape(-1, data_width * data_height).astype(np.float32)
img_test = img_test.reshape(-1, data_width * data_height).astype(np.float32)

print("training data set matrix dimensions: ", img_train.shape)
print("testing data set training dimensions: ", img_test.shape)

# generate the class labels
train_labels = np.repeat(numbers, no_train_data)[:, np.newaxis]
test_label = np.repeat(numbers, (total_data-no_train_data))[:, np.newaxis]

# convert them to floating data type
train_labels = train_labels.astype(np.float32)

# train with kNN
knn = cv2.ml.KNearest_create()
knn.train(img_train, cv2.ml.ROW_SAMPLE, train_labels)

# test the data
ret, result, neighbour, dist = knn.findNearest(img_test, k)

# display the results
print("k =", k)
matches = (result == test_label)
correct = np.count_nonzero(matches)
print("correct results: ", correct, "total results: ", result.size)
accuracy = correct*100.0/result.size
print("accuracy: ", accuracy)