import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

# Configuration parameters
DATA_PATH = 'MP_Data'
actions = np.array(['A', 'B', 'C'])
no_sequences = 30
sequence_length = 30
num_features = 63  # Update this based on your actual feature size

# Create label mapping
label_map = {label: num for num, label in enumerate(actions)}

# Load and preprocess data with dtype enforcement
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            npy_path = os.path.join(
                DATA_PATH, action, str(sequence), f"{frame_num}.npy"
            )
            try:
                frame_data = np.load(npy_path, allow_pickle=True).astype(np.float32)
            except:
                frame_data = np.zeros(num_features, dtype=np.float32)
            
            # Ensure consistent shape and type
            if frame_data.shape != (num_features,):
                frame_data = np.zeros(num_features, dtype=np.float32)
            
            window.append(frame_data)
        sequences.append(window)
        labels.append(label_map[action])

# Convert to properly typed numpy array
X = np.asarray(sequences, dtype=np.float32)
y = to_categorical(labels).astype(int)

# Verify data shape and type
print(f"Data shape: {X.shape}, Data type: {X.dtype}")
print(f"Sample element:\n{X[0,0,:5]}")  # Print first 5 features of first frame

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=42
)

# Rest of the model setup remains the same...
# Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', 
         input_shape=(sequence_length, num_features)),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])

# Compile and train model
model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=200,
    validation_data=(X_test, y_test),
    callbacks=[TensorBoard(log_dir='Logs')]
)

# Save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')