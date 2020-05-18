import numpy as np
import base64
import sys


def base64_encode_image(a):
	#base64 encode the input NumPy array
	return base64.b64encode(a).decode("utf-8")

def base64_decode_image(a, dtype, shape):
	#if this is python 3 we need the extra step
	#serialized NumPy string as a byte object
	if sys.version_info.major == 3:
		a = bytes(a, encoding = 'utf-8')

	#convert the string to NumPy array using the supplied data
	#type and target shape
	a = np.frombuffer(base64.decodestring(a), dtype = dtype)
	a = a.reshape(shape)

	#return the decode image
	return a
