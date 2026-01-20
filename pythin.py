# 🧠 Training a Neural Network on the MNIST Dataset using TensorFlow and Keras
# 🎯 Goal: To train a model that can recognize handwritten digits (0–9)

# ------------------------------------------------------------
# 📦 Step 1: Importing the Required Libraries
# ------------------------------------------------------------
# Here we are importing the necessary packages to build and train our neural network.
# TensorFlow and Keras help us create and train deep learning models easily.
import tensorflow as tf
from tensorflow.keras.datasets import mnist              # Contains the MNIST handwritten digits dataset
from tensorflow.keras.models import Sequential           # Used to create a simple layer-by-layer model
from tensorflow.keras.layers import Dense, Flatten        # Layers for our neural network
from tensorflow.keras.utils import to_categorical         # Converts labels into one-hot encoded format

# ------------------------------------------------------------
# 🧩 Step 2: Loading and Preparing the Dataset
# ------------------------------------------------------------
# Here we load the MNIST dataset, which automatically splits into training and testing data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# We normalize the images so that pixel values (0–255) are converted into 0–1 range.
# This helps the model train faster and more efficiently.
x_train, x_test = x_train / 255.0, x_test / 255.0

# Next, we convert the labels into one-hot encoded vectors.
# For example, the label ‘3’ becomes [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# ------------------------------------------------------------
# 🧱 Step 3: Building the Neural Network Model
# ------------------------------------------------------------
# Here we are creating a Sequential model, which means layers are added one after another.
model = Sequential([
    # The Flatten layer converts each 28x28 image into a 1D vector of 784 pixels.
    Flatten(input_shape=(28, 28)),
/

    # The output layer has 10 neurons — one for each digit (0–9).
    # We use Softmax to get probabilities for each class.
    Dense(10, activation='softmax')
])

# ------------------------------------------------------------
# ⚙️ Step 4: Compiling the Model
# ------------------------------------------------------------
# Here we specify how the model should learn.
# - Optimizer: 'adam' → adjusts weights efficiently
# - Loss function: 'categorical_crossentropy' → suitable for multi-class classification
# - Metric: 'accuracy' → to measure performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ------------------------------------------------------------
# 🎯 Step 5: Training the Model
# ------------------------------------------------------------
# Now we train the model on the training data.
# - epochs = 5 → model will go through the training data 5 times
# - batch_size = 32 → weights are updated after every 32 samples
model.fit(x_train, y_train, epochs=5, batch_size=32)

# ------------------------------------------------------------
# 📊 Step 6: Evaluating the Model
# ------------------------------------------------------------
# After training, we test the model on unseen test data to check its performance.
loss, acc = model.evaluate(x_test, y_test)

# ------------------------------------------------------------
# 🏁 Step 7: Displaying the Final Accuracy
# ------------------------------------------------------------
print(f"✅ Test Accuracy: {acc * 100:.2f}%")

# ------------------------------------------------------------
# 📚 Summary:
# ------------------------------------------------------------
# - We implemented a simple neural network to classify handwritten digits (0–9).
# - Steps included: Loading data → Normalizing → One-hot encoding → Building → Training → Evaluating.
# - The model usually achieves around 97–98% accuracy on the MNIST dataset.
