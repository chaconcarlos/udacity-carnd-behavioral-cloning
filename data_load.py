#!/usr/bin/env python
import os
import csv
import cv2
import glob
import math
import keras
import ntpath
import numpy as np
from keras.models               import Sequential
from keras.optimizers           import Adam
from keras.layers.core          import Dense, Lambda, Activation, Flatten, Dropout
from keras.regularizers         import l2
from keras.layers.pooling       import MaxPooling2D
from keras.preprocessing.image  import ImageDataGenerator
from keras.layers.convolutional import Convolution2D, Cropping2D
from sklearn.model_selection    import train_test_split

IMAGES_PATH_NAME       = "IMG"
CSV_FILENAME           = "driving_log.csv"
CSV_FIELD_CENTER_IMAGE = 0
CSV_FIELD_LEFT_IMAGE   = 1
CSV_FIELD_RIGHT_IMAGE  = 2
CSV_FIELD_STEERING     = 3
CSV_IMAGE_COUNT        = 3
STEERING_CORRECTION    = 0.1
TEST_DATA_SIZE_FACTOR  = 0.3
VALID_DATA_SIZE_FACTOR = 0.5
CROP_TOP               = 70
CROP_BOTTOM            = 25
LEARNING_RATE          = 0.001

# Print iterations progress
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

# Gets the line count in the given file.
def get_line_count(filename):
	line_count = 0
	with open(filename) as f:
		line_count = sum(1 for _ in f)
	return line_count

def load_data_entry(features, labels, csv_row, data_path, use_lateral_images = True):
	steering_center_angle = float(csv_row[CSV_FIELD_STEERING])
	image_range           = 1
	images_path           = os.path.join(data_path, IMAGES_PATH_NAME)

	if (use_lateral_images):
		image_range = CSV_IMAGE_COUNT

	for i in range(image_range):
		image_name     = ntpath.basename(csv_row[i]) 
		image          = cv2.imread(os.path.join(images_path, image_name))
		image          = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image          = image[CROP_TOP : image.shape[0] - CROP_BOTTOM, : , ]
		steering_angle = steering_center_angle

		if (i == CSV_FIELD_LEFT_IMAGE):
			steering_angle = steering_center_angle + STEERING_CORRECTION
		elif (i == CSV_FIELD_RIGHT_IMAGE):
			steering_angle = steering_center_angle - STEERING_CORRECTION

		features.append(image)
		labels.append(steering_angle)

# Loads the CSV file with the information from the generated data
def load_data(data_path, line_limit = -1, use_lateral_images = True):
	csv_filename = os.path.join(data_path, CSV_FILENAME)
	row_count    = get_line_count(csv_filename)
	file         = open(csv_filename)
	reader       = csv.reader(file)
	current_row  = 0
	features     = []
	labels       = []

	print("Loading data from generated images on {}...".format(csv_filename))

	if (line_limit == -1):
		line_limit = row_count

	for row in reader:
		load_data_entry(features, labels, row, data_path, use_lateral_images)
		current_row += 1
		printProgressBar(current_row, line_limit, prefix = "  Loading progress: ")
		if (line_limit == current_row):
			break

	print("  Loaded {} entries from {}.".format(current_row, csv_filename))
	print("  Total features loaded: {}".format(len(features)))

	image_shape = features[0].shape

	return np.array(features), np.array(labels), image_shape

# Builds the model. Based on the NVIDIA model.
def build_model(image_shape):
	#total_crop_y = CROP_TOP + CROP_BOTTOM
	#input_shape  = (image_shape[0] - total_crop_y, image_shape[1], image_shape[2])

	print ("Building model for images with shape: {}...".format(image_shape))

	model = Sequential()
	#model.add(Cropping2D(cropping=((CROP_TOP, CROP_BOTTOM), (0,0)), input_shape = input_shape))
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = image_shape))
	# Model chooses how to convert Color channels. As suggested by: https://chatbotslife.com/teaching-a-car-to-drive-himself-e9a2966571c5
	model.add(Convolution2D(1, 1, 1, border_mode = 'same', init ='glorot_uniform'))
	# Layer 1. Convolution 2D, 3 Filters, Kernel Size: 5x5. 
	model.add(Convolution2D(3, 5, 5, border_mode = 'valid', init ='glorot_uniform', W_regularizer=l2(0.01)))
	model.add(Activation('elu'))
	model.add(MaxPooling2D((2, 2), border_mode='valid'))
	# Layer 2. Convolution 2D, 24 Filters, Kernel Size: 5x5. 
	model.add(Convolution2D(24, 5, 5, border_mode = 'valid', init ='glorot_uniform'))
	model.add(Activation('elu'))
	model.add(MaxPooling2D((2, 2), border_mode='valid'))
	# Layer 3. Convolution 2D, 36 Filters, Kernel Size: 5x5. 
	model.add(Convolution2D(36, 5, 5, border_mode = 'valid', init ='glorot_uniform'))
	model.add(Activation('elu'))
	model.add(MaxPooling2D((2, 2), border_mode='valid'))
	# Layer 4. Convolution 2D, 48 Filters, Kernel Size: 3x3. 
	model.add(Convolution2D(48, 3, 3, border_mode = 'valid', init ='glorot_uniform'))
	model.add(Activation('elu'))
	model.add(MaxPooling2D((2, 2), border_mode='valid'))
	
	# Layer 5. Convolution 2D, 64 Filters, Kernel Size: 3x3. 
	#model.add(Convolution2D(64, 3, 3, border_mode = 'valid', init ='glorot_uniform'))
	#model.add(Activation('elu'))
	#model.add(MaxPooling2D((2, 2), border_mode='valid'))
	
	# Layer 6. Flatten.
	model.add(Flatten())
	# Layer 7. Fully Connected.
	model.add(Dense(1164, init='uniform'))
	model.add(Activation('elu'))
	# Layer 8. Fully Connected.
	model.add(Dense(100, init='uniform'))
	model.add(Activation('elu'))
	# Layer 9. Fully Connected.
	model.add(Dense(50, init='uniform'))
	model.add(Activation('elu'))
	# Layer 10. Fully Connected.
	model.add(Dense(10, init='uniform'))
	model.add(Activation('elu'))
	# Layer 10. Fully Connected.
	model.add(Dense(1, activation = 'linear'))

	model.summary()

	model.compile(optimizer = Adam(lr=LEARNING_RATE), loss='mse')

	return model

def train_model(model, X_train, y_train):
	model.fit(X_train, y_train, nb_epoch = 7, validation_split = VALID_DATA_SIZE_FACTOR, shuffle = True, verbose = 1)
	model.save("model.h5")

features, labels, image_shape    = load_data("data/", 10, True)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = TEST_DATA_SIZE_FACTOR,  random_state = 42)

model = build_model(image_shape)
train_model(model, X_train, y_train)

metrics = model.evaluate(X_test, y_test)

print (metrics)

#for metric_i in range(len(model.metrics_names)):
#    metric_name  = model.metrics_names[metric_i]
#    metric_value = metrics[metric_i]
#    print('{}: {}'.format(metric_name, metric_value))