import numpy as np
import pandas as pd
import scipy.io as mat
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.lines as mlines

def get_train_data(file: str):
    """
    Load and process training data from a .mat file.

    Parameters:
    file (str): The path to the .mat file.

    Returns:
    tuple: A tuple containing the x and y data arrays.
    """
    data = mat.loadmat(file)
    ETdata = data["ETdata"]
    mtype = ETdata.dtype
    ndata = {n: ETdata[n][0, 0] for n in mtype.names}
    data_headline = ndata['pos'][0]
    data_raw = ndata['pos']
    pdata = pd.DataFrame(data_raw, columns=data_headline)
    df = pd.DataFrame(pdata)
    x = pdata.iloc[:, 3:5].values
    y = pdata.iloc[:, 5].values
    return x, y

def show_colored_stage(x, y):
    """
    Visualize the stages of eye-tracking data with colors.

    Parameters:
    x (ndarray): The array containing the x coordinates.
    y (ndarray): The array containing the stage labels.
    """
    points = x[:, np.newaxis, :]
    lines = np.concatenate([points[:-1], points[1:]], axis=1)

    # Normalize y to the range [0, 1]
    y_norm = (y - y.min()) / (y.max() - y.min())

    # Use viridis colormap
    cmap = plt.get_cmap('viridis')
    line_colors = cmap(y_norm)

    lc = mcoll.LineCollection(lines, colors=line_colors, linewidths=2)

    _, ax = plt.subplots()
    ax.add_collection(lc)
    ax.autoscale()

    # Create a custom legend
    labels = ['Fixations', 'Saccades', 'Smooth pursuits', 'Post-Saccadic Oscillations', 'Glissades', 'Undefined?']
    legend_elements = [mlines.Line2D([], [], color=cmap((int(list(set(y))[i]) - 1) / (len(set(y)) - 1)), label=labels[int(list(set(y))[i]) - 1]) for i in range(len(set(y)))]
    ax.legend(handles=legend_elements, loc='best')

    plt.show()

def get_xy_vel(coordinates: np.ndarray):
    """
    Calculate the velocity in the x and y directions.

    Parameters:
    coordinates (ndarray): The array containing the coordinates.

    Returns:
    ndarray: The array containing the velocities in the x and y directions.
    """
    velocity = (coordinates[1:] - coordinates[:-1]) / 2e-3
    return velocity

def get_magnitude_vel(coordinates: np.ndarray):
    """
    Calculate the magnitude of velocity.

    Parameters:
    coordinates (ndarray): The array containing the coordinates.

    Returns:
    ndarray: The array containing the magnitudes of the velocities.
    """
    velocity = (coordinates[1:] - coordinates[:-1]) / 2e-3
    velocity = (velocity[:, 0] ** 2 + velocity[:, 1] ** 2) ** 0.5
    return velocity

def open_list_of_files(files_to_load):
    """
    Load and combine training data from multiple .mat files.

    Parameters:
    files_to_load (list): A list of paths to the .mat files.

    Returns:
    tuple: A tuple containing the combined x and y data arrays.
    """
    samples = []
    labels = []
    for my_file in files_to_load:
        sam, lab = get_train_data(my_file)
        samples.extend(sam)
        labels.extend(lab)
    samples = np.array(samples)
    labels = np.array(labels)
    print('Number of samples at the end:', len(samples))
    return samples, labels

def make_sequences(samples, labels, sequence_dim=100, sequence_lag=1, sequence_attributes=2):
    """
    Create sequences of data samples and corresponding labels.

    Parameters:
    samples (ndarray): The array containing the samples.
    labels (ndarray): The array containing the labels.
    sequence_dim (int): The dimension of each sequence. Default is 100.
    sequence_lag (int): The lag between sequences. Default is 1.
    sequence_attributes (int): The number of attributes in each sample. Default is 2.

    Returns:
    tuple: A tuple containing the sequences and corresponding labels arrays.
    """
    nsamples = []
    nlabels = []
    for i in range(0, samples.shape[0] - sequence_dim, sequence_lag):
        nsample = np.zeros((sequence_dim, sequence_attributes))
        for j in range(i, i + sequence_dim):
            nsample[j - i, 0] = samples[j, 0]
            nsample[j - i, 1] = samples[j, 1]
        nlabel = labels[i + sequence_dim // 2]
        nsamples.append(nsample)
        nlabels.append(nlabel)

    samples = np.array(nsamples)
    labels = np.array(nlabels)
    labels = np.where(labels >= 2, 2, 1)
    return samples, labels