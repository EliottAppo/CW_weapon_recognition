from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import cv2


def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes are integers, convert them to floats -- this
	# is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# compute the area of the bounding boxes and grab the indexes to sort
	# (in the case that no probabilities are provided, simply sort on the
	# bottom-left y-coordinate)
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = y2

	# if probabilities are provided, sort on them instead
	if probs is not None:
		idxs = probs

	# sort the indexes
	idxs = np.argsort(idxs)

	# keep looping while some indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the index value
		# to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of the bounding
		# box and the smallest (x, y) coordinates for the end of the bounding
		# box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have overlap greater
		# than the provided overlap threshold
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked
	return boxes[pick].astype("int"),probs[pick]

def sliding_window(image, step, ws):
	# slide a window across the image
	for y in range(0, image.shape[0] - ws[1], step):
		for x in range(0, image.shape[1] - ws[0], step):
			# yield the current window
			yield (x, y, image[y:y + ws[1], x:x + ws[0]])

def image_pyramid(image, scale=1.5, minSize=(224, 224)):
    # yield the original image
	yield image
	# keep looping over the image pyramid
	while True:
		# compute the dimensions of the next image in the pyramid
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		# yield the next image in the pyramid
		yield image

model = ResNet50(weights="imagenet", include_top=True)


# initialize two lists, one to hold the ROIs generated from the image
# pyramid and sliding window, and another list used to store the
# (x, y)-coordinates of where the ROI was in the original image
rois1 = []
locs = []

def detection_with_prediction(file_path, min_conf = 0.5):
    """Renvoie l'image avec un rectangle autour des armes.
    On démultiplie notre image en pyramide d'images de différentes tailles,
    Puis chaque image est découpé avec une fenetre glissante.
    Chaque image obtenue passe par la prédiction.
    Puis on garde que les rectangles où on prédit une arme.
    Enfin on fusionne ces rectangles si ils se superposent
    
    ROI_SIZE indique la taille des rectangles, il faut connaitre la taille de l'objet
    que l'on recherche sur l'image si on veut que cette fonction soit efficace"""
    # initialize variables used for the object detection procedure
    WIDTH = 600
    orig = cv2.imread(file_path)
    orig = imutils.resize(orig, width=WIDTH)
    (H, W) = orig.shape[:2]
    PYR_SCALE = 1.5
    WIN_STEP = 16
    INPUT_SIZE = (224, 224)
    ROI_SIZE = (H//2,W//2)


    # loop over the image pyramid
    pyramid = image_pyramid(orig, scale=PYR_SCALE, minSize=ROI_SIZE)
    for image in pyramid:
        # determine the scale factor between the *original* image
        # dimensions and the *current* layer of the pyramid
        scale = W / float(image.shape[1])
        # for each layer of the image pyramid, loop over the sliding
        # window locations
        for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
            # scale the (x, y)-coordinates of the ROI with respect to the
            # *original* image dimensions
            x = int(x * scale)
            y = int(y * scale)
            w = int(ROI_SIZE[0] * scale)
            h = int(ROI_SIZE[1] * scale)
            # take the ROI and preprocess it so we can later classify
            # the region using Keras/TensorFlow
            roi = cv2.resize(roiOrig, INPUT_SIZE)
            roi = img_to_array(roi)
            roi = preprocess_input(roi)
            # update our list of ROIs and associated coordinates
            rois1.append(roi)
            locs.append((x, y, x + w, y + h))

    # convert the ROIs to a NumPy array
    rois = np.array(rois1, dtype="float32")
    # classify each of the proposal ROIs using ResNet
    preds = model.predict(rois)
    # decode the predictions and initialize a dictionary which maps class
    # labels (keys) to any ROIs associated with that label (values)
    preds = imagenet_utils.decode_predictions(preds, top=1)
    #On prend que les armes
    pred2=[]
    for x in preds:
        if x[0][1] in set(["revolver","rifle","assault_rifle"]):
            pred2.append(x)
    preds = pred2
    labels={}
    print('Loop over prediction')
    # loop over the predictions
    for (i, p) in enumerate(preds):
        # grab the prediction information for the current ROI
        (imagenetID, label, prob) = p[0]
        # filter out weak detections by ensuring the predicted probability
        # is greater than the minimum probability
        if prob >= min_conf:
            # grab the bounding box associated with the prediction and
            # convert the coordinates
            box = locs[i]
            # grab the list of predictions for the label and add the
            # bounding box and probability to the list
            L = labels.get(label, [])
            L.append((box, prob))
            labels[label] = L
    print('Done')
    clone = orig.copy()
    for label in labels.keys():
        # extract the bounding boxes and associated prediction
        # probabilities, then apply non-maxima suppression
        boxes = np.array([p[0] for p in labels[label]])
        proba = np.array([p[1] for p in labels[label]])
        boxes,proba = non_max_suppression(boxes, proba)
        # loop over all bounding boxes that were kept after applying
        # non-maxima suppression
        print(proba)
        print(boxes)
        #if there is many weapons :
        # for items in zip(boxes,proba):
        #     startX, startY, endX, endY = list(items[0])
        #     # draw the bounding box and label on the image
        #     cv2.rectangle(clone, (startX, startY), (endX, endY),
        #          (0, 255, 0), 2)
        #     y = startY - 10 if startY - 10 > 10 else startY + 10
        #     cv2.putText(clone, label +str(items[1]), (startX, y),
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        #     # show the output after apply non-maxima suppression
        #if there is 1 weapon :
        p = proba[0]
        r = 0
        for i in range(len(proba)):
            if proba[i] > p :
                 r = i
        startX, startY, endX, endY = boxes[r]
        cv2.rectangle(clone, (startX, startY), (endX, endY),
                  (0, 255, 0), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(clone, label +str(p), (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    cv2.imshow("After", clone)
    cv2.waitKey(0)
    
def detection_with_prediction_for_test(file_path, min_conf = 0.5):
    """Meme chose que detection_with_prediction mais retourne le découpage de l'arme"""
    # initialize variables used for the object detection procedure
    WIDTH = 600
    orig = cv2.imread(file_path)
    orig = imutils.resize(orig, width=WIDTH)
    (H, W) = orig.shape[:2]
    PYR_SCALE = 1.5
    WIN_STEP = 16
    INPUT_SIZE = (224, 224)
    ROI_SIZE = (H//2,W//2)


    # loop over the image pyramid
    pyramid = image_pyramid(orig, scale=PYR_SCALE, minSize=ROI_SIZE)
    for image in pyramid:
        # determine the scale factor between the *original* image
        # dimensions and the *current* layer of the pyramid
        scale = W / float(image.shape[1])
        # for each layer of the image pyramid, loop over the sliding
        # window locations
        for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
            # scale the (x, y)-coordinates of the ROI with respect to the
            # *original* image dimensions
            x = int(x * scale)
            y = int(y * scale)
            w = int(ROI_SIZE[0] * scale)
            h = int(ROI_SIZE[1] * scale)
            # take the ROI and preprocess it so we can later classify
            # the region using Keras/TensorFlow
            roi = cv2.resize(roiOrig, INPUT_SIZE)
            roi = img_to_array(roi)
            roi = preprocess_input(roi)
            # update our list of ROIs and associated coordinates
            rois1.append(roi)
            locs.append((x, y, x + w, y + h))

    # convert the ROIs to a NumPy array
    rois = np.array(rois1, dtype="float32")
    # classify each of the proposal ROIs using ResNet
    preds = model.predict(rois)
    # decode the predictions and initialize a dictionary which maps class
    # labels (keys) to any ROIs associated with that label (values)
    preds = imagenet_utils.decode_predictions(preds, top=1)
    #On prend que les armes
    pred2=[]
    for x in preds:
        if x[0][1] in set(["revolver","rifle","assault_rifle"]):
            pred2.append(x)
    preds = pred2
    labels={}
    print('Loop over prediction')
    # loop over the predictions
    for (i, p) in enumerate(preds):
        # grab the prediction information for the current ROI
        (imagenetID, label, prob) = p[0]
        # filter out weak detections by ensuring the predicted probability
        # is greater than the minimum probability
        if prob >= min_conf:
            # grab the bounding box associated with the prediction and
            # convert the coordinates
            box = locs[i]
            # grab the list of predictions for the label and add the
            # bounding box and probability to the list
            L = labels.get(label, [])
            L.append((box, prob))
            labels[label] = L
    print('Done')
    clone = orig.copy()
    for label in labels.keys():
        # extract the bounding boxes and associated prediction
        # probabilities, then apply non-maxima suppression
        boxes = np.array([p[0] for p in labels[label]])
        proba = np.array([p[1] for p in labels[label]])
        print(boxes,type(boxes))
        boxes,proba = non_max_suppression(boxes, proba)
        # loop over all bounding boxes that were kept after applying
        # non-maxima suppression
        print(proba)
        print(boxes)
        #if there is many weapons :
        # for items in zip(boxes,proba):
        #     startX, startY, endX, endY = list(items[0])
        #     # draw the bounding box and label on the image
        #     cv2.rectangle(clone, (startX, startY), (endX, endY),
        #          (0, 255, 0), 2)
        #     y = startY - 10 if startY - 10 > 10 else startY + 10
        #     cv2.putText(clone, label +str(items[1]), (startX, y),
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        #     # show the output after apply non-maxima suppression
        #if there is 1 weapon :
        p = proba[0]
        r = 0
        for i in range(len(proba)):
            if proba[i] > p :
                 r = i
        return boxes[r]
