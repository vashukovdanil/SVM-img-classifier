import os
from sklearn.datasets import fetch_lfw_people


# Графика
import matplotlib as mpl
import matplotlib.pyplot as plt

# Работа с данными
import pandas as pd
import numpy as np

# Работа с растровой графикой
from PIL import Image, ImageDraw, ImageFont

# Обработка изображений
from skimage.feature import hog
from skimage.color import rgb2gray

# Обучение
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Оценка
from sklearn.metrics import roc_curve, auc, accuracy_score





def get_image(id, root_dir):
    # Открывает изображение в папке и возвращает его в качестве массива
    file = f'{id}.png'
    path = os.path.join(root_dir, file)
    img = Image.open(path)
    img = img.resize((200,200), Image.ANTIALIAS)
    img_arr = np.array(img)

    return img_arr

def create_features(img):
    # Делает изображение серым и сглаживает его до одной строки
    gray_img = rgb2gray(img)
    flat_features = hog(gray_img, block_norm='L2-Hys', pixels_per_cell=(16, 16))
    return flat_features

def create_feature_matix():
    # Пробегает по всем папкам-классам и
    # создаёт матрицу по всех изображениям
    
    names = ["abd", "chst"]
    matrix = []
    labels = []

    for set_name in names:
        root = f"train/{set_name}/"
        f_counts = len([name for name in os.listdir(root)])

        for i in range(1, f_counts+1):
            img = get_image(i, root)
            img_ft = create_features(img)
            matrix.append(img_ft)
            labels.append(set_name)
    
    matrix = pd.DataFrame(matrix)
    return labels, matrix

def get_accuracy(y_test, y_pred):
    # Выводит точность модели
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: ', accuracy)

def train_model():
    # Тренировка модели. Возвращает обученную модель.
    y, X = create_feature_matix()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    get_accuracy(y_test, y_pred)

    return svm


def classify_img(model, name, path):
    # Классификация собственных изображений
    img = get_image(name, path)
    X = create_features(img).reshape(-1, 1).T
    
    # Класс и его вероятность
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    path = os.path.join(path, f"{name}.png")
    img = Image.open(path)
    draw = ImageDraw.Draw(img)
    
    # Текст на изображении
    txt = f'Class: {y_pred[0]}\nP(abd): {np.round(y_prob[0][0]*100, 2)}%\nP(chst): {np.round(y_prob[0][1]*100, 2)}%' 
    draw.text((0, 0), txt)
    
    img.show()


if __name__ == "__main__":
    svm = train_model()
    path = input("Ведите путь к изображению: ")
    name = input("Введите имя изображения (png): ")
    classify_img(svm, name, path)