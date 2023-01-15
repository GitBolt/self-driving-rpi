import os
from sklearn.model_selection import train_test_split
from utils import *
from pre_process import *
from model import create_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

path = 'data'
data = import_data_info(path)

# Visualize and balance data
data = balance_data(data, display=True)

# Prepare data for processing
images_path, steerings = load_data(path, data)

# Split data for training and validation
x_train, x_val, y_train, y_val = train_test_split(
    images_path,
    steerings,
    test_size=0.2,
    random_state=10
)
print('Total training images:', len(x_train))
print('Total validation images:', len(x_val))

model = create_model()

history = model.fit(generate_data(x_train, y_train, 100, 1),
                    steps_per_epoch=100,
                    epochs=10,
                    validation_data=generate_data(x_val, y_val, 50, 0),
                    validation_steps=50)

model.save('model.h5')
print('Model saved!')

# Plot results
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
