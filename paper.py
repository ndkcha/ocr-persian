import numpy as np
import cv2
import os
import math

# path to the data set
dir_train_data = os.fsencode("training_data")

# number of digits to train
n = 10
# total number of samples
total_data = 1000
# number of samples to use as training data (testing data will be calculated accordingly)
no_train_data = 700
# k in k-Nearest Neighbour (number of neighbours to consider for classification)
k = 5
# C for SVM
c = 2.67
# Gamma for SVM
gamma = 5.383

# labels for the training data classes
numbers = np.arange(n)
# counters to keep track of the training and testing data sets
noOfTraining = 0
noOfTesting = 0

# training samples (to be updated from the image inputs)
# we have total of 88 features from the input
img_train = np.empty([no_train_data*n, 88])
img_test = np.empty([(total_data-no_train_data)*n, 88])

print("Loading digits...")
# iterate through the directory to get digit samples
for digit_samples in numbers:
    i = 0
    print("Loading #" + str(digit_samples))
    # iterate through the image samples
    for digits in os.listdir(os.fsdecode(dir_train_data) + "/" + str(digit_samples)):
        # indexes of each features
        ff = 0
        sf = 16
        tf = 32
        # feature vector
        feature_vector = np.zeros(88)
        # get the sample image
        img = cv2.imread(os.fsdecode(dir_train_data) + "/" + str(digit_samples) + "/" + os.fsdecode(digits), 0)
        # resize it
        r_img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_CUBIC)
        # reduce noise (canny edge detection) (test case: 1)
        # c_img = cv2.Canny(r_img, 100, 200)
        # remove the erosion (test case: 2)
        kernel = np.ones((5, 5), np.uint8)
        c_img = cv2.erode(r_img, kernel, iterations=1)
        i_img = cv2.bitwise_not(c_img)

        # divide the image into 5x5 sub matrices
        div_img = np.array(i_img).reshape((-1, 5, 5))
        # iterate through the matrices to extract the features
        for sub_matrix in div_img:
            # calculate the zeros in the matrix (first feature)
            zeros= np.where(sub_matrix == 255)
            # add the first feature to the vector
            feature_vector[ff] = zeros[0].size
            if feature_vector[ff] == 0:
                continue
            # derive the coordinates of the zeros
            co_zeros = np.transpose(zeros)
            # calculate the mean distance from origin (2nd feature) and the mean angle from the origin (3rd feature)
            mean_distance = 0.0
            mean_angle = 0.0
            for coordinates in co_zeros:
                mean_distance += ((coordinates[0]**2) + (coordinates[1]**2))**0.5
                mean_angle += coordinates[0] == 0 and 90 or math.degrees(math.atan(coordinates[1]/coordinates[0]))
            # add the second feature to the vector
            feature_vector[sf] = mean_distance/float(feature_vector[ff])
            feature_vector[tf] = mean_angle/float(feature_vector[ff])
            ff += 1
            sf += 1
            tf += 1

        # add fourier transform (fourth feature)
        ff_img = np.fft.rfft(np.array(i_img.reshape(-1)))
        fft_img = ((ff_img.real ** 2) + (ff_img.imag ** 2)) ** 0.5
        feature_vector[48:] = fft_img[:40]

        # convert it to array
        if i < no_train_data:
            img_train[noOfTraining] = feature_vector
            noOfTraining += 1
        else:
            img_test[noOfTesting] = feature_vector
            noOfTesting += 1
        i += 1

print(img_train[0])

img_train = img_train.reshape(-1, 88).astype(np.float32)
img_test = img_test.reshape(-1, 88).astype(np.float32)

print("training data set matrix dimensions: ", img_train.shape)
print("testing data set training dimensions: ", img_test.shape)

# generate the class labels
train_labels = np.repeat(numbers, no_train_data)[:, np.newaxis]
test_label = np.repeat(numbers, (total_data-no_train_data))[:, np.newaxis]

# convert them to floating data type
train_labels_float = train_labels.astype(np.float32)

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