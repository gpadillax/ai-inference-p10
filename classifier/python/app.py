# Copyright 2022 IBM All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implements a Flask server that classifies a
JPEG image as one of a thousand categories using a ResNet model.
"""
import io
import json
from typing import List

import flask
import labels
import numpy as np
import onnxruntime as ort
import resources
from PIL import Image
from prediction import Prediction
from preprocess import preprocess
import utils

app = flask.Flask(__name__)
INFERENCE_SESSION = ort.InferenceSession(
    resources.ONNX_MODEL_FILE_PATH, providers=["CPUExecutionProvider"]
)
CLASS_LABELS: List[labels.Label] = labels.get_class_labels(
    label_file=resources.LABEL_FILE_PATH
)


@app.route("/alive", methods=["GET"])
def is_alive() -> flask.Response:
    """Returns 200 OK for a readiness probe"""
    return flask.Response(status=200)


@app.route("/infer", methods=["POST"])
def infer() -> flask.Response:
    """
    Runs the inference on the provided JPG file and returns a json of
    the top 5 predictions where the score is greater than zero.
    """
    # Load and preprocess the image (scale/resize/normalize)
    input_image = Image.open(io.BytesIO(flask.request.data))
    image_array = preprocess(input_image)

    # Make the single image into a batch
    batch = np.expand_dims(image_array, axis=0)

    # Make Predictions.
    # There is only one input (named "data") for this model, and with
    # a batch size of 1, there is only one row of scores as output.
    output = INFERENCE_SESSION.run([], {"data": batch})[0].flatten()
    scores = utils.softmax(output)

    top5 = [
        Prediction(label=CLASS_LABELS[p], score=round(float(scores[p]), 3))
        for p in np.argsort(-scores)[:5]
    ]

    # Send response json
    return flask.Response(
        json.dumps({"predictions": [p.dict() for p in top5 if p.score > 0]}),
        content_type="application/json",
        status=200,
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
