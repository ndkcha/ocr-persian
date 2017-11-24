import cv2
import numpy as np
import os
import sys

# directory pointing to training data
dir_train_data = os.fsencode("training_data")

# number of digits to recognize
n = 10
# number of total data set
total_data = 1000
# number of training data set (selected from the total data set)
no_train_data = 700
# value of k for kNN method
k = 5
# value of C for SVM method
c = 2.67
# value of Gamma for SVM method
gamma = 5.383
# image size for computing feature space
# all the images will be resized to this size
dim = 34
img_size = (dim, dim)
# flags for the affine transformation (in order to rotate the image)
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
# bin size of histogram (for HoG method)
bin_n = 16

# array of digits to recognize
numbers = np.arange(n)
# tracking variables to store data in memory
noOfTraining = 0
noOfTesting = 0

# named assignments to store the data sets in memory
img_train = np.empty([no_train_data*n, 64])
img_test = np.empty([(total_data-no_train_data)*n, 64])


# deskew the image using the moments. rotate to the appropriate angle.
# param: image - input image from the data set
# return: image - the properly rotated image
def deskewMoments(image):
    # calculate the moments of the image
    m = cv2.moments(image)
    # check the variance of the mass distribution w.r.t vertical axis
    # return the image if the distribution is appropriate
    if abs(m['mu02']) < 1e-2:
        return image.copy()
    # determine transformation matrix (based on skew value) from the covariance and variance of the mass distribution
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * img_size[0] * skew], [0, 1, 0]])
    # perform transformation
    image = cv2.warpAffine(image, M, img_size, flags=affine_flags)
    return image


# deskew the image by determining the angle. rotate to the appropriate angle.
# param: image - input image from the data set
# return: image - the properly rotated image
def deskewAngle(image):
    # store the coordinates for the values greater than 0
    coords = np.column_stack(np.where(image > 0))
    # generate the rotated bounding box for all the coordinates
    angle = cv2.minAreaRect(coords)[-1]

    # correct the angle value, because the above function returns in range [-90, 0)
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # get the height and the width
    (h, w) = image.shape[:2]
    # calculate the center of the image
    center = (w // 2, h // 2)
    # determine the transformation matrix (based on the rotation values)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # perform transformation
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return image


# find the histogram of gradient descriptor
def hog(image):
    # compute the horizontal and vertical gradients
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    # convert the coordinate system to polar, that gives magnitude and angle.
    mag, ang = cv2.cartToPolar(gx, gy)
    # determine the bins
    bins = np.int32(bin_n*ang/(2*np.pi))
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    # calculate the histogram
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist


# load the data set and perform the pre processing and feature extraction
print("Loading digits...")
for digit_samples in numbers:
    i = 0
    for digits in os.listdir(os.fsdecode(dir_train_data) + "/" + str(digit_samples)):
        print("Loading digits # %d%%\r" % ((digit_samples * 10) + i / 100), end="")
        img = cv2.imread(os.fsdecode(dir_train_data) + "/" + str(digit_samples) + "/" + os.fsdecode(digits), 0)
        r_img = cv2.resize(img, img_size)
        # threshold the image. In a way, it inverts the image.
        thresh_img = cv2.threshold(r_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # deskew the image
        d_img = deskewMoments(thresh_img)
        # histogram of gradient descriptor
        h_img = hog(d_img)
        # divide between the training and testing data.
        if i < no_train_data:
            img_train[noOfTraining] = np.array(h_img).reshape(64)
            noOfTraining += 1
        else:
            img_test[noOfTesting] = np.array(h_img).reshape(64)
            noOfTesting += 1
        i += 1


# make sure that the data is properly shaped.
img_train = img_train.reshape(-1, 64).astype(np.float32)
img_test = img_test.reshape(-1, 64).astype(np.float32)

print("training data set matrix dimensions: ", img_train.shape)
print("testing data set training dimensions: ", img_test.shape)

# generate the labels
train_labels = np.repeat(numbers, no_train_data)[:, np.newaxis]
test_label = np.repeat(numbers, (total_data-no_train_data))[:, np.newaxis]

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
# perform kNN and SVM on test data
knn_result = knn.findNearest(img_test, k)[1]
svm_result = svm.predict(img_test)[1]

# determine the accuracy of the results and print it.
print("kNN : k =", k, "SVM : c =", c, "Gamma =", gamma)
knn_matches = (knn_result == test_label)
knn_correct = np.count_nonzero(knn_matches)
svm_matches = (svm_result == test_label)
svm_correct = np.count_nonzero(svm_matches)
print("total results: ", knn_result.size, "correct results - kNN: ", knn_correct, "SVM: ", svm_correct)
knn_acc= knn_correct*100.0/knn_result.size
svm_acc = svm_correct*100.0/svm_result.size
print("accuracy - kNN: ", knn_acc, "SVM: ", svm_acc)