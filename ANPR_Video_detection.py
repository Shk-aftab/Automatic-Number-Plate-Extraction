import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from local_utils import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import glob

def load_model(path):
	try:
		path = splitext(path)[0]
		with open(path +'.json', 'r') as json_file:
			model_json = json_file.read()
		model = model_from_json(model_json, custom_objects={})
		model.load_weights(path +'.h5')
		print('Loading model successfull')
		return model
	except Exception as e:
		print(e)
            
wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)

mobileNet_json_path = 'MobileNets_character_recognition.json'
mobileNet_weight_path = 'License_character_recognition_weight.h5'
mobileNet_classes_path = 'license_character_classes.npy'

def load_mobileNet_model(json_path,weight_path,classes_path):

	json_file = open(json_path, 'r')
	loaded_model_json = json_file.read()
	r_model = model_from_json(loaded_model_json)
	r_model.load_weights(weight_path)
	print("[INFO] Model loaded successfully...")

	labels = LabelEncoder()
	labels.classes_ = np.load(classes_path)
	print("[INFO] Labels loaded successfully...")

	return r_model,labels

recognition_model,class_labels = load_mobileNet_model(mobileNet_json_path,mobileNet_weight_path,mobileNet_classes_path)

def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction


Dmin = 256
Dmax = 608

video_path = 'bike_number_plate_video.mp4'

try:
	vid = cv2.VideoCapture(int(video_path))
except:
	vid = cv2.VideoCapture(video_path)

while True:
	return_value, frame = vid.read()
	if return_value:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		vehicle = frame / 255
		ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
		side = int(ratio * Dmin)
		bound_dim = min(side, Dmax)
		_,LpImg,_,cor = detect_lp(wpod_net, vehicle, bound_dim,lp_threshold=0.5)

		plate_image = cv2.convertScaleAbs(LpImg[0],alpha=(255.0))

		# conver to grayscale
		gray = cv2.cvtColor(plate_image,cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray,(7,7),0)

		#Applied inversed thresh_binary where the pixel value less than threshold will be converted to 255 and vice versa
		binary = cv2.threshold(blur,180,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

		## Applied dilation
		kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
		thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

		cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		sorted_contours = sorted(cont, key=lambda ctr: cv2.boundingRect(ctr)[0])

		# creat a copy version "test_roi" of plat_image to draw bounding box
		test_roi = plate_image.copy()

		# Initialize a list which will be used to append charater image
		crop_characters = []

		# define standard width and height of character
		digit_w, digit_h = 30, 60

		for c in sorted_contours:
			(x,y,w,h) = cv2.boundingRect(c)
			ratio = h/w
			if 1<=ratio<=3.5:
				if h/plate_image.shape[0]>=0.5:
					cv2.rectangle(test_roi,(x,y),(x+w,y+h),(0,255,0),2)
					cv2.imshow('kuch',test_roi)
					cv2.waitKey(0)

					curr_num = thre_mor[y-5:y+h+5,x-5:x+w+5]
					curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
					_, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
					crop_characters.append(curr_num)

		print("Detect {} letters...".format(len(crop_characters)))

		final_string = '' 
		for i, character in enumerate(crop_characters):
			title = np.array2string(predict_from_model(character,recognition_model,class_labels))
		final_string+=title.strip("'[]")

		print(final_string)

