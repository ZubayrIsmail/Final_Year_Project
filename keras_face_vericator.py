#Taken from https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/
from numpy import expand_dims
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from keras_vggface import VGGFace
from keras_vggface.utils import preprocess_input, decode_predictions
from mtcnn.mtcnn import MTCNN
import tensorflow as tf

resnet50_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
vgg16_model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3), pooling='avg')
senet50_model = VGGFace(model='senet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def extract_face(filename, required_size=(224, 224)):
	'''
	get faces from given image using MTCNN
	:param filename:
	:param required_size:
	:return:
	'''
	# load image from file
	pixels = pyplot.imread(filename)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array


def get_embeddings(filename, model):
	'''
	gets image embeddings using given model
	:param filename:
	:param model:
	:return:
	'''
	# extract faces
	face = [extract_face(filename)]
	# convert into an array of samples
	sample = asarray(face, 'float32')
	# prepare the face for the model, e.g. center pixels
	sample = preprocess_input(sample, version=2)
	# perform prediction
	yhat = model.predict(sample)
	return yhat


#
def is_match(score, thresh=0.4):
	'''
	determine if a candidate face is a match for a known face
	:param score:
	:param thresh:
	:return:
	'''
	if score <= thresh:
		print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
	else:
		print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))

def validate(ground_truth_file_path, candidate_file_path, model):
	'''
	Validate a candidate against an image using a given model
	:param ground_truth_file_path:
	:param candidate_file_path:
	:param model: defaults to Resnet50
	:return: score between 0 and 1 (lower is better)
	'''
	# get embeddings file filenames
	candidate_embedding = get_embeddings(candidate_file_path, model)
	ground_truth_embedding = get_embeddings(ground_truth_file_path, model)
	return cosine(ground_truth_embedding, candidate_embedding)


