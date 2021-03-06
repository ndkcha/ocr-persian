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
data_width = 40
# resultant height of the sample image
data_height = 34
# C for SVM
c = 2.67
# Gamma for SVM
gamma = 5.383

# labels for the training data classes
numbers = np.arange(n)
# counters to keep track of the training and testing data sets
noOfTraining = 0
noOfTesting = 0

# training samples (to be updated from the image inputs
img_train = np.empty([no_train_data*n, data_width*data_height])
img_test = np.empty([(total_data-no_train_data)*n, data_width*data_height])

# iterate through the directory to get digit samples
for digit_samples in numbers:
    i = 0
    # iterate through the image samples
    for digits in os.listdir(os.fsdecode(dir_train_data) + "/" + str(digit_samples)):
        print("Loading digits # %d%%\r" % ((digit_samples * 10) + i / 100), end="")
        # get the sample image
        img = cv2.imread(os.fsdecode(dir_train_data) + "/" + str(digit_samples) + "/" + os.fsdecode(digits), 0)
        # resize it
        img = cv2.resize(img, (data_width, data_height), interpolation=cv2.INTER_CUBIC)
        # reduce noise (canny edge detection) (test case: 1)
        # img = cv2.Canny(img, 50, 300)
        # remove the erosion (test case: 2)
        kernel = np.ones((3,3),np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
        # invert it
        img = cv2.bitwise_not(img)
        # convert it to array
        if i < no_train_data:
            img_train[noOfTraining] = np.array(img).reshape(data_width * data_height)
            noOfTraining += 1
        else:
            img_test[noOfTesting] = np.array(img).reshape(data_width * data_height)
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
train_labels_float = train_labels.astype(np.float32)

print("# training model...\r", end="")

# train with kNN
knn = cv2.ml.KNearest_create()
knn.train(img_train, cv2.ml.ROW_SAMPLE, train_labels_float)

# train with SVM
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(c)
svm.setGamma(gamma)
svm.train(img_train, cv2.ml.ROW_SAMPLE, train_labels)

print("# verifying model...\r", end="")

# test the data
ret, knn_result, neighbour, dist = knn.findNearest(img_test, k)
svm_result = svm.predict(img_test)[1]

# display the results
print("kNN : k =", k, "SVM : c =", c, "Gamma =", gamma)
knn_matches = (knn_result == test_label)
knn_correct = np.count_nonzero(knn_matches)
svm_matches = (svm_result == test_label)
svm_correct = np.count_nonzero(svm_matches)
print("total results: ", knn_result.size, "correct results - kNN: ", knn_correct, "SVM: ", svm_correct)
knn_acc= knn_correct*100.0/knn_result.size
svm_acc = svm_correct*100.0/svm_result.size
print("accuracy - kNN: ", knn_acc, "SVM: ", svm_acc)