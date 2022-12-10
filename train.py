from sklearn.model_selection import train_test_split
import os
from utils import *

print('Setting up...')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import data
path = 'data'
data = importDataInfo(path)
print(data.head())

# Visualize and balance data
data = balanceData(data, display=True)

# Prepare data for processing
imagesPath, steerings = loadData(path, data)

# Split data for training and validation
x_train, x_val, y_train, y_val = train_test_split(imagesPath, steerings,
                                                  test_size=0.2, random_state=10)
print('Total training images:', len(x_train))
print('Total validation images:', len(x_val))

# Augment data

# Preprocess

# Create model
model = createModel()

# Train model
history = model.fit(dataGen(x_train, y_train, 100, 1),
                    steps_per_epoch=100,
                    epochs=10,
                    validation_data=dataGen(x_val, y_val, 50, 0),
                    validation_steps=50)

# Save model
model.save('model.h5')
print('Model saved!')

# Plot results
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
