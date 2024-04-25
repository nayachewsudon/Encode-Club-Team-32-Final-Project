import sys
import json
import ast
from transformers import pipeline
from PIL import Image, ImageDraw
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def root():
    return "root"

@app.route("/object-detection/<image_name>")
def object_detection(image_name):

    checkpoint = "google/owlv2-base-patch16-ensemble"
    detector = pipeline(model=checkpoint, task="zero-shot-object-detection")

    # filename = sys.argv[1]
    filename = image_name
    outputFileName = f"{filename.split('.')[0]}_detection.png"
    

    image = Image.open(filename)

    labels = "Bone fracture"

    predictions = detector(
        image,
        candidate_labels=labels,
    )

    # Image
    draw = ImageDraw.Draw(image)
    a=1
    for prediction in predictions:
        box = prediction["box"]
        label = prediction["label"]
        score = prediction["score"]

        xmin, ymin, xmax, ymax = box.values()
        draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
        draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="black")
        a+=1
        if a > 1:
            break

    # Text
    i=1
    for prediction in predictions:
        label = prediction["label"]
        score = prediction["score"] * 100
        suffix = ''
        if 11 <= (i % 100) <= 13:
            suffix = 'th'
        else:
            suffix = ['th', 'st', 'nd', 'rd', 'th'][min(i % 10, 4)]
        # print(f"The word {label} is the {i}{suffix} most related to the image with a confidence of {score:.2f}%")
        output= {"outputFileName": outputFileName, "diagnose" : f"The word {label} is the {i}{suffix} most related to the image with a confidence of {score:.2f}%"}
        i+=1
        if i > 1:
            break

    image.save(outputFileName)
    # output.append({ "outputFileName": outputFileName })
    # output = { "outputFileName": outputFileName }

    # print(json.dumps(output))
    return jsonify(output), 200

    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3001)

