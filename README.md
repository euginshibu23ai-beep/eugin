from google.colab import files
uploaded = files.upload()

from IPython.display import display, Javascript
from google.colab.output import eval_js
import cv2
import numpy as np
import base64

def take_photo(filename='photo.jpg', quality=0.8):
    js = '''
      async function takePhoto(quality) {
        const div = document.createElement('div');
        const capture = document.createElement('button');
        capture.textContent = 'Capture';
        div.appendChild(capture);

        const video = document.createElement('video');
        video.style.display = 'block';
        div.appendChild(video);
        document.body.appendChild(div);

        const stream = await navigator.mediaDevices.getUserMedia({video: true});
        video.srcObject = stream;
        await video.play();

        // Resize for button placement
        google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

        await new Promise((resolve) => capture.onclick = resolve);

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        stream.getVideoTracks()[0].stop();
        div.remove();

        return canvas.toDataURL('image/jpeg', quality);
      }
      takePhoto({quality: %f});
    ''' % quality
    data = eval_js(js)
    header, encoded = data.split(',', 1)
    img = np.frombuffer(base64.b64decode(encoded), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    cv2.imwrite(filename, img)
    return filename  

     take_photo('me.jpg')

     import os
import urllib.request

# Define the URL for the cascade classifier file
cascade_url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
cascade_filename = 'haarcascade_frontalface_default.xml'

# Check if the file exists, if not, download it
if not os.path.exists(cascade_filename):
    print(f"Downloading {cascade_filename}...")
    urllib.request.urlretrieve(cascade_url, cascade_filename)
    print(f"{cascade_filename} downloaded.")

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier(cascade_filename)

# Check if the cascade classifier loaded successfully
if face_cascade.empty():
    print("Error: Face cascade classifier not loaded correctly.")
else:
    print("Face cascade classifier loaded successfully.")



    # Read input image
img = cv2.imread('me.jpg')

# Check if the image was loaded successfully
if img is None:
    print("Error: Could not load image.")
else:
    print(f"Image shape: {img.shape}") # Print image shape for debugging
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if gray is None or gray.size == 0:
        print("Error: Grayscale image is empty.")
    else:
        print(f"Grayscale image shape: {gray.shape}") # Print grayscale image shape for debugging
        # Display grayscale image for debugging
        import matplotlib.pyplot as plt
        plt.imshow(gray, cmap='gray')
        plt.title("Grayscale Image")
        plt.axis('off')
        plt.show()

        # Check if the cascade classifier is loaded
        if face_cascade.empty():
            print("Error: Face cascade classifier not loaded correctly.")
        else:
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))

            # Limit faces to 4
            faces = faces[:4]

            # Blue color in BGR
            blue_color = (255, 0, 0)
            black_color = (0, 0, 0)

            # Draw rectangles and add colored name and relationship labels
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), blue_color, 2)

                # Define starting positions for the text - moved 15 pixels higher than before
                name_label_pos = (x, y - 45)
                name_value_pos = (x + 70, y - 45)  # 70 pixels to the right, same y

                relation_label_pos = (x, y - 25)
                relation_value_pos = (x + 120, y - 25)  # 120 pixels to the right, same y

                # Put labels in black (changed from white)
                cv2.putText(img, 'Name:', name_label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, black_color, 2)
                cv2.putText(img, 'Relationship:', relation_label_pos,  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Put values in blue (BGR)
                cv2.putText(img, 'Eugin Shibu', name_value_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, blue_color, 2)
                cv2.putText(img, 'son ', relation_value_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, blue_color, 2)

            # Show image with updated labels
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()



            print(f"Number of faces detected: {len(faces)}")


            take_photo('Elizebath(Mother).jpg')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


img = cv2.imread('Elizebath(Mother).jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))

# Limit faces to 4
faces = faces[:4]

# Green color in BGR
green_color = (0, 255, 0)

# Draw rectangles and add colored name and relationship labels
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), green_color, 2)

    # Define starting positions for the text with more upward offset
    name_label_pos = (x, y - 45)         # Moved up by additional 15 pixels
    name_value_pos = (x + 70, y - 45)    # Match name label y

    relation_label_pos = (x, y - 25)     # Moved up by additional 15 pixels
    relation_value_pos = (x + 120, y - 25)  # Match relationship label y

    # Put labels in black
    cv2.putText(img, 'Name:', name_label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, 'Relationship:', relation_label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Put values in green
    cv2.putText(img, 'elizebath', name_value_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, green_color, 2)
    cv2.putText(img, '  Mother', relation_value_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, green_color, 2)

# Show image with updated labels
import matplotlib.pyplot as plt
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

print(f"Number of faces detected: {len(faces)}")


20take_photo('MotherMe(Group).jpg')

import cv2
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('MotherMe(Group).jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))


names = ["elizebath", "eugin"]
relationships = [" mother", "son"]
colors = [(255, 0, 0), (0, 255, 0)]  # Blue for you, Green for mother


for i, (x, y, w, h) in enumerate(faces[:2]):
    color = colors[i]
    name = names[i]
    relation = relationships[i]

    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    # Increase y-offset by 15 pixels for more space
    cv2.putText(img, "Name:", (x, y - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, name, (x + 70, y - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(img, "Relationship:", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img, relation, (x + 140, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


print(f"Faces detected: {len(faces)} (labeled first two with black prefix and colored info)")

take_photo('shibu(Father).jpg')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read input image
img = cv2.imread('shibu(Father).jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))

# Limit faces to 4
faces = faces[:4]

# Yellow color in BGR
yellow_color = (0, 255, 255)

# Draw rectangles and add colored name and relationship labels
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), yellow_color, 2)

    # Define starting positions for the text
    name_label_pos = (x, y -55)
    name_value_pos = (x + 70, y - 55)  # 70 pixels to the right of label

    relation_label_pos = (x, y - 30)
    relation_value_pos = (x + 120, y - 30)  # 120 pixels to the right of label

    # Put labels in white
    cv2.putText(img, 'Name:', name_label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, 'Relationship:', relation_label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Put values in yellow
    cv2.putText(img, 'shibu', name_value_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, yellow_color, 2)
    cv2.putText(img, ' father', relation_value_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, yellow_color, 2)

# Show image with updated labels
import matplotlib.pyplot as plt
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

print(f"Number of faces detected: {len(faces)}")
take_photo('fatherrMe(Group).jpg')

import cv2
import matplotlib.pyplot as plt


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('fatherrMe(Group).jpg')  # Replace with your actual image path
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

names = ["eugin", "shibu"]
relationships = ["son", "father"]
colors = [(255, 0, 0), (0, 255, 255)]  # Blue for you, Yellow for your brother

for i, (x, y, w, h) in enumerate(faces[:2]):
    color = colors[i]
    name = names[i]
    relation = relationships[i]

    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    # Increase y-offset by 15 pixels for more space
    cv2.putText(img, "Name:", (x, y - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, name, (x + 70, y - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(img, "Relationship:", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, relation, (x + 140, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

print(f"Faces detected: {len(faces)} (labeled first two with black prefix and colored info)")

