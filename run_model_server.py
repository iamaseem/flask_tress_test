#import area
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import imagenet_utils
import numpy as np
import settings
import helpers
import redis
import time
import json

#connect to Redis SERVER
db = redis.StrictRedis(host = settings.REDIS_HOST, port = settings.REDIS_PORT,
						db = settings.REDIS_DB)

def classify_process():
	print("Loading model...")
	model = ResNet50(weights = "imagenet")
	print("Model Loaded")

	while True:
		queue = db.lrange(settings.IMAGE_QUEUE, 0,
							settings.BATCH_SIZE - 1)
		imageIDs = []
		batch = None

		for q in queue:
			q = json.loads(q.decode("utf-8"))
			image = helpers.base64_decode_image(q["image"],
						settings.IMAGE_DTYPE,
						(1, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH,
						settings.IMAGE_CHANS))

			if batch is None:
				batch = image

			else:
				batch = np.vstack([batch, image])

			imageIDs.append(q["id"])

		if len(imageIDs) > 0:
			print("* Batch size: {}".format(batch.shape))
			preds = model.predict(batch)
			result = imagenet_utils.decode_predictions(preds)

			for (imageIDs, resultSet) in zip(imageIDs, result):
				output = []
				for (imageIDs, label, prob)	in resultSet:
					r = {"label": label, "probability": float(prob)}
					output.append(r)

				db.set(imageIDs, json.dumps(output))

			db.ltrim(settings.IMAGE_QUEUE, len(imageIDs) - 1)

		time.sleep(settings.SERVER_SLEEP)

if __name__ == __main__:
	classify_process()
