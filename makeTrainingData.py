# %%
from sklearn.preprocessing import LabelBinarizer
from modules import *
import glob
import os
import numpy as np

# Read all files
directory = r"Resources/EyeData"
full_directory_files = glob.glob(directory + r"/*")
# %% Getting all training data
for i in range(len(full_directory_files) - 1, -1, -1):
    x, y = get_train_data(full_directory_files[i])
    
    # if there is a data that was not described in the paper
    # Delete it
    if max(set(y)) > 4:
        os.remove(full_directory_files[i])
        full_directory_files.pop(i)
    else: 
        print(set(y))
        show_colored_stage(x, y)

# %% Make training sequences
s, l = open_list_of_files(full_directory_files)
s = get_xy_vel(s)

sequence_dim = 100

x, y = make_sequences(s, l)
print(f"Shape after conversion: {x.shape}; {y.shape}")
val_split = round(x.shape[0] * .1)
x_train = x[: x.shape[0] - val_split]
y_train = y[: x.shape[0] - val_split]
x_test = x[x.shape[0] - val_split:]
y_test = y[x.shape[0] - val_split:]

# %% Labeling all data and normalize to [0, 1]

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

mean = np.mean(x)
std_dev = np.std(x)
x_train = (x_train - mean) / std_dev
x_test = (x_test - mean) / std_dev

# %% Changing data to one hot encoding
y_train_sudo = np.zeros((y_train.shape[0], 2))
for i in range(y_train.shape[0]):
    y_train_sudo[i, y_train[i, 0]] = 1
y_train = y_train_sudo

y_test_sudo = np.zeros((y_test.shape[0], 2))
for i in range(y_test.shape[0]):
    y_test_sudo[i, y_test[i, 0]] = 1
y_test = y_test_sudo
# %% Save all training and test data for AI Cooking
data = {"x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test}

np.savez("Resources/train.npz", **data)
# %%
