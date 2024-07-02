# %%
from keras import layers, models, optimizers, regularizers as reg
from keras.callbacks import EarlyStopping
from modules import *
import keras
import matplotlib.pyplot as plt
import numpy as np

# %%
data = np.load("Resources/train.npz")
x_train = data['x_train']
x_test = data['x_test']
y_train = data['y_train']
y_test = data['y_test']

# %%
"""Build the model according to the paper"""

inputShape = x_train.shape[1:]
print('inputShape:',inputShape)
model = models.Sequential(name = 'CNN_EyePhaseDetector')
model.add(layers.Input(inputShape))
model.add(layers.Conv1D(32, 3, kernel_regularizer = reg.L1L2()))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.Dropout(0.2))

model.add(layers.Conv1D(64, 3, padding="same", kernel_regularizer = reg.L1L2()))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Conv1D(128, 3, padding="same", kernel_regularizer = reg.L1L2()))
model.add(layers.Activation("relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Flatten())
#model.add(Dense(128, activation='sigmoid'))
model.add(layers.Dense(64, activation='sigmoid', kernel_regularizer = reg.L1L2()))
model.add(layers.Dense(2, activation='softmax'))
#model.add(Dense(3, activation='softmax'))

model.summary()

# %% Conditions for model's patience

earlystopping = EarlyStopping(
    monitor = "val_loss",
    min_delta = 0.0006,
    patience = 50,
    verbose = True,
    restore_best_weights = True,
    mode = 'min'
)
# %% training model based on the learning rate

lrs = [1e-4, 5e-5, 1.5e-5]

for idx, lr in enumerate(lrs):

    model.compile(optimizer = optimizers.Adam(learning_rate = lr, beta_1 = 0.975), metrics = ['accuracy'], loss = 'categorical_crossentropy')
    csv_logger = keras.callbacks.CSVLogger(f"Resources/EyePhaseClassifier({idx}).csv")
    history = model.fit(x_train, y_train, epochs = 200, batch_size = 128, validation_data = [x_test, y_test], callbacks = [csv_logger, earlystopping], shuffle = True)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
# %% 
"""WARNING: IF YOU PLAN TO SAVE THE MODEL, MAKE A BACKUP OF THE PREVIOUS ONE FIRST"""
model.save(r'Resources/EyePhaseClassifier.keras')
# %%

model = models.load_model(r'Resources/EyePhaseClassifier.keras')
model.summary()
# %%
from sklearn.metrics import confusion_matrix
import numpy as np

predictor = model.predict(x_train, batch_size = 1024)
y_pred = np.argmax(predictor, axis = 1)
y_truth = np.argmax(y_train, axis = 1)


cm = confusion_matrix(y_truth, y_pred, normalize = 'pred')
# %%
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


# %%
