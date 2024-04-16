import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the paths to your image directories
with_mask_dir = './images/with_mask'
without_mask_dir = './images/without_mask'

# Set up the image data generator
datagen = ImageDataGenerator(rescale=1./255)
# Load the images from the directories
with_mask_data = datagen.flow_from_directory(with_mask_dir, target_size=(224, 224), batch_size=32, class_mode='binary')
without_mask_data = datagen.flow_from_directory(without_mask_dir, target_size=(224, 224), batch_size=32, class_mode='binary')

# Create the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(with_mask_data, epochs=10, validation_data=without_mask_data)

# Save the model
model.save('mask_classification_model.h5')