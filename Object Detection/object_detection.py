import sys
from transformers import pipeline
from PIL import Image, ImageDraw

checkpoint = "google/owlv2-base-patch16-ensemble"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection")

filename = sys.argv[1]
image = Image.open(filename)

labels = "Bone fracture"

predictions = detector(
    image,
    candidate_labels=labels,
)


# Image
draw = ImageDraw.Draw(image)
for prediction in predictions:
    box = prediction["box"]
    label = prediction["label"]
    score = prediction["score"]

    xmin, ymin, xmax, ymax = box.values()
    draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
    draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="black")

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
    print(f"The word {label} is the {i}{suffix} most related to the image with a confidence of {score:.2f}%")
    i+=1

image.save(f"{filename.split('.')[0]}_detection.png")