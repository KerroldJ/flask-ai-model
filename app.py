from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import mediapipe as mp
from function import *

app = Flask(__name__)
try:
    with open("model.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("model.h5")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading error: {str(e)}")
    exit(1)

actions = ['A','B','C', 'ILoveYou', 'Hello', 'Thankyou', 'Please', 'No', 'Me' ]
threshold = 0.5

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

sequence = []
sentence = []
predictions = []
accuracy = []

@app.route('/predict', methods=['POST'])
def predict():
    global sequence, sentence, predictions, accuracy

    try:
        # Ensure frame is provided
        if 'frame' not in request.files:
            print("❌ No frame provided in request")
            return jsonify({"error": "No frame provided"}), 400

        file = request.files['frame']
        image = np.frombuffer(file.read(), dtype=np.uint8)
        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

        if frame is None:
            print("❌ Failed to decode image")
            return jsonify({"error": "Failed to decode image"}), 400

        # Crop and process frame
        cropframe = frame[40:400, 0:300]  # Ensure this cropping logic is correct

        # Perform MediaPipe detection
        image, results = mediapipe_detection(cropframe, hands)
        if results is None:
            print("❌ MediaPipe detection failed")
            return jsonify({"error": "MediaPipe detection failed"}), 500
        
        print("MediaPipe detection complete")

        if results.multi_hand_landmarks:
            # Extract keypoints
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]  # Maintain a 30-frame window

            if len(sequence) == 30:
                # Ensure sequence is properly shaped for model input
                input_data = np.expand_dims(sequence, axis=0)

                # Make prediction
                res = model.predict(input_data)[0]
               

                # Handle empty or incorrect model output
                if len(res) == 0:
                    print("❌ Model returned an empty prediction")
                    return jsonify({"error": "Model output is empty"}), 500

                max_index = np.argmax(res)
                if max_index >= len(actions):
                    print(f"❌ Model predicted index {max_index}, out of range for actions list")
                    return jsonify({"error": "Model prediction index out of range"}), 500

                predictions.append(max_index)
                current_action = actions[max_index]
                current_accuracy = res[max_index]

                # Update sentence logic
                if current_accuracy > threshold:
                    if len(predictions) > 10:
                        # Check for consistent predictions
                        if np.all(np.array(predictions[-10:]) == max_index):
                            if len(sentence) == 0 or (sentence[-1] != current_action):
                                sentence.append(current_action)
                                accuracy.append(f"{current_accuracy * 100:.1f}%")

                                # Keep only last 3 items
                                sentence = sentence[-3:]
                                accuracy = accuracy[-3:]

                # Prepare response
                response = {
                    "action": current_action,
                    "accuracy": float(current_accuracy * 100),  # Convert to percentage
                    "sentence": sentence,
                    "accuracy_list": accuracy
                }
                return jsonify(response)

        print("⚠️ No hand landmarks detected")
        return jsonify({
            "action": None,
            "accuracy": 0.0,
            "sentence": [],
            "accuracy_list": []
        })

    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()  # Print full error stack trace
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
