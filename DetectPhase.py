    # %%
# TODO:
# - Understand what the hell is the input
# - What the fuck is the CNN doing
# - What type of shit does it produces
# - And how the fuck to evaluate the goddamn model

# Eye movement stage
# - Fixations
# - Saccades
# - Smooth pursuits
# - Post-Saccadic Oscillations
# - Glissades

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import glob
import threading
import numpy as np

np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.lines as mlines
from keras import layers, models, optimizers, regularizers as reg
from keras.callbacks import EarlyStopping
import keras
import scipy.io.matlab as mat

directory = r"/mnt/C26889126889067F/Coding/Python/NeuralNetwork/Resources/EyeData"
full_directory_files = glob.glob(directory + r"/*")
print(full_directory_files)
# %%
"""This cells defines necessary functions"""
def get_train_data(file: str):
    data = mat.loadmat(file)
    ETdata = data["ETdata"]
    mtype = ETdata.dtype
    ndata = {n: ETdata[n][0, 0] for n in mtype.names}
    data_headline = ndata['pos']
    data_headline = data_headline[0]
    data_raw = ndata['pos']
    pdata = pd.DataFrame(data_raw, columns = data_headline)
    df = pd.DataFrame(pdata)
    x = pdata.iloc[:, 3: 5].values
    y = pdata.iloc[:, 5].values
    return x, y

def show_colored_stage(x, y):
    points = x[:, np.newaxis, :]
    lines = np.concatenate([points[: -1], points[1:]], axis = 1)

    # Normalize y to the range [0, 1]
    y_norm = (y - y.min()) / (y.max() - y.min())

    # Use viridis colormap
    cmap = plt.get_cmap('viridis')
    line_colors = cmap(y_norm)

    lc = mcoll.LineCollection(lines, colors = line_colors, linewidths = 2)

    _, ax = plt.subplots()
    ax.add_collection(lc)
    ax.autoscale()

    # Create a custom legend
    labels = ['Fixations', 'Saccades', 'Smooth pursuits', 'Post-Saccadic Oscillations', 'Glissades', 'Undefined?']
    legend_elements = [mlines.Line2D([], [], color=cmap((int(list(set(y))[i]) - 1) / (len(set(y))-1)), label=labels[int(list(set(y))[i]) - 1]) for i in range(0, len(set(y)))]
    ax.legend(handles=legend_elements, loc='best')

    plt.show()

def get_xy_vel(coordinates: np.ndarray): 
    # Calculate velocity in x and y direction seperately
    velocity = (coordinates[1:] - coordinates[: -1]) / 2e-3
    return velocity

def get_magnitude_vel(coordinates: np.ndarray): 
    # Calculate magnitude of velocity
    velocity = (coordinates[1:] - coordinates[: -1]) / 2e-3
    velocity = (velocity[:, 0] ** 2 + velocity[:, 1] ** 2) ** .5
    return velocity

def open_list_of_files(files_to_load):
    samples = []
    labels =[]
    for my_file in files_to_load:
        sam,lab = get_train_data(my_file)
        samples.extend(sam)
        labels.extend(lab)
    samples = np.array(samples)
    labels = np.array(labels)
    print('Number of samples at the end:',len(samples))
    return samples, labels
    
def make_sequences(samples, labels, sequence_dim = 100, sequence_lag = 1, sequence_attributes = 2):
    nsamples = []
    nlabels = [] 
    for i in range(0,samples.shape[0]-sequence_dim,sequence_lag):
        nsample = np.zeros((sequence_dim,sequence_attributes))
        for j in range(i,i+sequence_dim):
            nsample[j-i,0] = samples[j,0]
            nsample[j-i,1] = samples[j,1]
        nlabel = labels[i+sequence_dim//2]
        #print("Sample",nsample)
        #print("Label",nlabel)
        nsamples.append(nsample)
        nlabels.append(nlabel)
        
    samples = np.array(nsamples)
    labels = np.array(nlabels)
    labels = np.where(labels >= 2, 2, 1)
    return samples,labels 
    
# %%
import os
for i in range(len(full_directory_files) - 1, -1, -1):
    x, y = get_train_data(full_directory_files[i])
    if max(set(y)) > 4:
        os.remove(full_directory_files[i])
        full_directory_files.pop(i)
    else: 
        print(set(y))
        show_colored_stage(x, y)

# %%
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

# %% 
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

mean = np.mean(x)
std_dev = np.std(x)
x_train = (x_train - mean) / std_dev
x_test = (x_test - mean) / std_dev

# %%
y_train_sudo = np.zeros((y_train.shape[0], 2))
for i in range(y_train.shape[0]):
    y_train_sudo[i, y_train[i, 0]] = 1
y_train = y_train_sudo

y_test_sudo = np.zeros((y_test.shape[0], 2))
for i in range(y_test.shape[0]):
    y_test_sudo[i, y_test[i, 0]] = 1
y_test = y_test_sudo
# %% Build the model

inputShape = (sequence_dim, 2)
#inputShape = (x.shape)
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
model.add(layers.Dropout(0.2))
model.add(layers.Conv1D(128, 3, padding="same", kernel_regularizer = reg.L1L2()))
model.add(layers.Activation("relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
#model.add(Dense(128, activation='sigmoid'))
model.add(layers.Dense(64, activation='sigmoid', kernel_regularizer = reg.L1L2()))
model.add(layers.Dense(2, activation='softmax'))
#model.add(Dense(3, activation='softmax'))

model.summary()
# %%
earlystopping = EarlyStopping(
    monitor = "val_loss",
    min_delta = 0.0006,
    patience = 200,
    verbose = 2,
    restore_best_weights = True,
    mode = 'min'
)


model.compile(optimizer = optimizers.Adam(learning_rate = .0000001), metrics = ['accuracy'], loss = 'categorical_crossentropy')
csv_logger = keras.callbacks.CSVLogger(r"Resources/EyePhaseClassifier.csv")
history = model.fit(x_train, y_train, epochs = 200, batch_size = 128, validation_data = [x_test, y_test], callbacks = [csv_logger, earlystopping], shuffle = True)

# %% Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# %%
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
import dlib
import cv2
import numpy as np
import threading
import time
import pygame
import sys
import math

class VideoCapture:
    def __init__(self, index = 0):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 540)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, args=(), daemon = True)
        self.thread.start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()
        
    def read(self):
        return self.ret, self.frame
        

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()
        
def closeHoles(image):
    # Close all the loops formed in iris
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25), (-1, -1))
    dialate = cv2.dilate(image, element, iterations = 1)
    
    # Reverse back to iris original size
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11), (-1, -1))
    erode = cv2.erode(dialate, element, iterations = 1)
    return erode
    
        
def extract_rotated_rectangle(image, rect: tuple):
    centerX, centerY, width, angle, lowerHeight, upperHeight = rect
    
    rotation_matrix = cv2.getRotationMatrix2D((centerX, centerY), angle, scale=1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    x, y = centerX - width // 2, centerY - upperHeight
    cropped_roi = rotated_image[y:y + upperHeight + lowerHeight, x:x + width]

    return cropped_roi

def calc_coordinates(*pt):
    calc_angle = lambda right, left: np.arctan2((right[1] - left[1]), (right[0] - left[0]))
    centerX, centerY = (pt[0][0] + pt[3][0]) // 2, (pt[0][1] + pt[3][1]) // 2
    rotAngle = calc_angle(pt[3], pt[0])
    
    def find_dist(right, left, rotAngle):
        lowerDist = math.dist(right, left)
        angle = calc_angle(right, left)
        trueDist = np.sin(rotAngle - angle) * lowerDist
        return trueDist
    
    distLower1 = -find_dist(pt[5], pt[0], rotAngle)
    distLower2 = -find_dist(pt[4], pt[0], rotAngle)
    lowerHeight = math.ceil((distLower1 + distLower2) / 2)
    
    distUpper1 = find_dist(pt[1], pt[0], rotAngle)
    distUpper2 = find_dist(pt[2], pt[0], rotAngle)
    upperHeight = math.ceil((distUpper1 + distUpper2) / 2)
    return centerX, centerY + 2, round(math.dist(pt[0], pt[3])), round(180 / np.pi * rotAngle), lowerHeight, upperHeight

max_value = 255
max_value_H = 360//2

low_H = 93
low_S = 40
low_V = 0
high_H = 134
high_S = 85
high_V = 133

cv2.namedWindow("Threshold")
cv2.createTrackbar("low_H", "Threshold" , low_H, max_value_H, lambda a:...)
cv2.createTrackbar("high_H", "Threshold" , high_H, max_value_H, lambda a:...)
cv2.createTrackbar("low_S", "Threshold" , low_S, max_value, lambda a:...)
cv2.createTrackbar("high_S", "Threshold" , high_S, max_value, lambda a:...)
cv2.createTrackbar("low_V", "Threshold" , low_V, max_value, lambda a:...)
cv2.createTrackbar("high_V", "Threshold" , high_V, max_value, lambda a:...)
def irisCropper(eye) -> tuple[int, int]:
    """This function calculates center of the iris using a multitude of techniques

    Args:
        eye (_type_): _description_

    Returns:
        tuple[int, int]: _description_  
    """
    
    min_H = cv2.getTrackbarPos("low_H", "Threshold")
    max_H = cv2.getTrackbarPos("high_H", "Threshold")
    min_S = cv2.getTrackbarPos("low_S", "Threshold")
    max_S = cv2.getTrackbarPos("high_S", "Threshold")
    min_V = cv2.getTrackbarPos("low_V", "Threshold")
    max_V = cv2.getTrackbarPos("high_V", "Threshold")

    
    eye_hsv = cv2.cvtColor(eye, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(eye_hsv, (7,7), 0)
    threshHSVImg = cv2.inRange(blur[:, :, 2], ( min_H), (max_H))
    threshHSVImg = closeHoles(threshHSVImg)

    
    # Cluster seperation wizardry bullshit
    seperateCluster = cv2.distanceTransform(threshHSVImg, cv2.DIST_C, 5)
    cv2.normalize(seperateCluster, seperateCluster, 0, 1, cv2.NORM_MINMAX)
    _, seperateCluster = cv2.threshold(seperateCluster, .2, 255, cv2.THRESH_BINARY)
    seperateCluster = seperateCluster.astype(np.uint8)

    contours, hierarchy = cv2.findContours(seperateCluster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Select contours with highest number of points
    selectedContour = max(contours, key = len)
    
    
    # Center calculation of centroids
    moments = cv2.moments(selectedContour)
    Cx = int(moments['m10'] // moments['m00'])
    Cy = int(moments['m01'] // moments['m00'])

    cv2.circle(eye, (Cx, Cy), 2, (0, 255, 0), cv2.FILLED)
    cv2.imshow("Threshold", threshHSVImg)
    cv2.imshow("Eye Center", eye)
    return (Cx, Cy)
    


linux_path = r"/mnt/C26889126889067F/Coding/Python/NeuralNetwork/Resources/dlib-models/shape_predictor_68_face_landmarks_GTX.dat"
win_path = r"D:\Coding\Python\NeuralNetwork\Resources\dlib-models\shape_predictor_68_face_landmarks.dat"

cap = cv2.VideoCapture(r"/mnt/C26889126889067F/Coding/Python/NeuralNetwork/Resources/c808b974a3e9f73953ebafde2b91b679.mp4")
cnnDetector = dlib.cnn_face_detection_model_v1(r"/mnt/C26889126889067F/Coding/Python/NeuralNetwork/Resources/dlib-models/mmod_human_face_detector.dat")
detect = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(linux_path)

frame_cnt = 0
start = time.time()


while True:
    ret, frame = cap.read()
    if ret == False or (cv2.waitKey(20) & 0xFF == ord('k')):
        cv2.destroyAllWindows()
        cap.release()
        break

    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cnnFaces = cnnDetector(gray)
    for cnnFace in cnnFaces:
        rect = cnnFace.rect
        startX = rect.left()
        startY = rect.top()
        endX = rect.right()
        endY = rect.bottom()
        frame = frame[startY: endY, startX: endX]
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detect(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        choosen = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        rect = calc_coordinates(*choosen)
        eye = extract_rotated_rectangle(frame, rect)
        eye = cv2.resize(eye, (200, 80))
        
        cannyEye = irisCropper(eye)

        
        
    if time.time() - start >= 1:
        print(frame_cnt, end = '\r', flush = True)
        start = time.time()
        frame_cnt = 0
    else: frame_cnt += 1
    
    # cv2.imshow("Frame", frame)
    
# %% Build eye tracking
import dlib
import cv2
import numpy as np
import threading
import time
import pygame
import sys
import math

class VideoCapture:
    def __init__(self, index = 0):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 540)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, args=(), daemon = True)
        self.thread.start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()
        
    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()
        
class SCREEN(object):
    def __init__(self, screensz: tuple, buttonBorder: int) -> None:
        self.xmax, self.ymax = screensz[0], screensz[1]
        self.screen = pygame.display.set_mode((0, 0))
        self.buttonBorder = buttonBorder
        
    def draw_button(self, coordinates: tuple, string: str, string_rel: tuple) -> pygame.Rect:
        """Create buttons method"""
        x, y = coordinates[0], coordinates[1]
        x_rel, y_rel = string_rel[0], string_rel[1]

        
        if y + coordinates[-1] > self.buttonBorder:
            raise Warning(f"Exceeded predefined button border by user: {self.buttonBorder}")
        
        rect = pygame.Rect(*coordinates)
        pygame.draw.rect(self.screen, (0, 0, 255), rect)
        text = pygame.font.SysFont(None, 24).render(string, True, (255, 255, 255))
        self.screen.blit(text, (x + x_rel, y + y_rel))
        return rect
    
    def drawX(self, coordinates: tuple) -> None:
        """Draw X at specific coodinates"""
        x, y = coordinates[0], coordinates[1]
        if x > self.xmax: x = x - self.xmax
        if y > self.ymax: y = y - self.ymax
        if y < self.buttonBorder: y = self.buttonBorder
        
        pygame.draw.line(self.screen, (255, 0, 0), (x, y), (20 + x, 20 + y), 2)
        pygame.draw.line(self.screen, (255, 0, 0), (x, 20 + y), (20 + x, y), 2)
        
    def clear(self) -> None:
        """When invoke will clear the screen"""
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(0, self.buttonBorder, self.xmax, self.ymax - self.buttonBorder))


def extract_rotated_rectangle(image, rect: tuple):
    centerX, centerY, width, angle, lowerHeight, upperHeight = rect
    
    rotation_matrix = cv2.getRotationMatrix2D((centerX, centerY), angle, scale=1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    x, y = centerX - width // 2, centerY - upperHeight
    cropped_roi = rotated_image[y:y + upperHeight + lowerHeight, x:x + width]

    return cropped_roi

def calc_coordinates(*pt):
    calc_angle = lambda right, left: np.arctan((right[1] - left[1]) / (right[0] - left[0]))
    centerX, centerY = (pt[0][0] + pt[3][0]) // 2, (pt[0][1] + pt[3][1]) // 2
    rotAngle = calc_angle(pt[3], pt[0])
    
    def find_dist(right, left, rotAngle):
        lowerDist = math.dist(right, left)
        angle = calc_angle(right, left)
        trueDist = np.sin(rotAngle - angle) * lowerDist
        return trueDist
    
    distLower1 = -find_dist(pt[5], pt[0], rotAngle)
    distLower2 = -find_dist(pt[4], pt[0], rotAngle)
    lowerHeight = math.ceil((distLower1 + distLower2) / 2)
    
    distUpper1 = find_dist(pt[1], pt[0], rotAngle)
    distUpper2 = find_dist(pt[2], pt[0], rotAngle)
    upperHeight = math.ceil((distUpper1 + distUpper2) / 2)
    return centerX, centerY, round(math.dist(pt[0], pt[3])), round(180 / np.pi * rotAngle), lowerHeight, upperHeight
    
        
pygame.init()
info = pygame.display.Info()
screen_width, screen_height = 1920, 1080
screen = SCREEN((screen_width, screen_height), 30)


linux_path = r"/mnt/C26889126889067F/Coding/Python/NeuralNetwork/Resources/dlib-models/shape_predictor_68_face_landmarks.dat"
win_path = r"D:\Coding\Python\Neural Network\Resources\dlib-models\shape_predictor_68_face_landmarks.dat"
cap = VideoCapture(0)

cnnDetector = dlib.cnn_face_detection_model_v1(r"/mnt/C26889126889067F/Coding/Python/NeuralNetwork/Resources/dlib-models/mmod_human_face_detector.dat")
detect = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(linux_path)

frame_cnt = 0
start = time.time()
stop = False

Xrand = False
start_Xrand = time.time()

while True:
            
    quit_rect = screen.draw_button((0, 0, 100, 30), "Quit", (32, 9))
    start_eval = screen.draw_button((110, 0, 100, 30), "Start Eval", (14, 9))
    stop_eval = screen.draw_button((220, 0, 100, 30), "Stop Eval", (16, 9))

    mouse_pos = pygame.mouse.get_pos()
    if quit_rect.collidepoint(mouse_pos):
        if pygame.mouse.get_pressed()[0]:
            stop = True
            
    if start_eval.collidepoint(mouse_pos):
        if pygame.mouse.get_pressed()[0]:
            Xrand = True
    
    if stop_eval.collidepoint(mouse_pos):
        if pygame.mouse.get_pressed()[0]:
            Xrand = False
    
    if Xrand and time.time() - start_Xrand >= 1.5: 
        screen.clear()
        screen.drawX((np.random.randint(0, screen_width), np.random.randint(0, screen_height)))
        start_Xrand = time.time()
            
    pygame.display.flip()
    
    # Tracking frame
    if time.time() - start >= 1:
        start = time.time()
        screen.draw_button((700, 0, 100, 30), f"FPS: {frame_cnt}", (17, 9))
        frame_cnt = 0
    else: frame_cnt += 1
    
    # Display camera
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cnnFaces = cnnDetector(gray)
    for cnnFace in cnnFaces:
        rect = cnnFace.rect
        startX = rect.left()
        startY = rect.top()
        endX = rect.right()
        endY = rect.bottom()
        frame = frame[startY: endY, startX: endX]
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detect(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        choosen = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        rect = calc_coordinates(*choosen)
        eye = extract_rotated_rectangle(frame, rect)
        eye = cv2.resize(eye, (200, 100))
        print(eye.shape, end = '\r', flush = True)
        try: cv2.imshow("Eye", eye)
        except: pass
        for i in choosen:
            cv2.circle(frame, i, 1, (0, 255, 0), cv2.FILLED)
        
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    
    for event in pygame.event.get():
        if stop or event.type == pygame.QUIT or key == ord('k'):
            cv2.destroyAllWindows()
            cap.release()
            pygame.quit()
            sys.exit()
# %%