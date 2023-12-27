import cv2
import os
import pytesseract
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Path ke folder dengan gambar PNG
folder_path = 'pics'
# ...

# Inisialisasi list untuk menyimpan data gambar dan label
images = []
labels = []

# Fungsi untuk melakukan preprocessing gambar
def preprocess_image(image):
    # Konversi gambar ke skala abu-abu
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thresholding untuk meningkatkan kontras teks
    _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Normalisasi intensitas warna (optional)
    normalized_image = cv2.normalize(thresholded_image, None, 0, 255, cv2.NORM_MINMAX)

    return normalized_image

# Fungsi untuk mengekstrak teks dari gambar menggunakan Tesseract OCR
def extract_text(image):
    # Hapus karakter non-alfanumerik (opsional)
    text = pytesseract.image_to_string(image, config='--psm 8')
    return text

# Loop melalui setiap gambar dalam folder
for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        # Baca gambar
        img_path = os.path.join(folder_path, filename)
        
        # Pengecekan file gambar sebelum membacanya
        if os.path.exists(img_path):
            image = cv2.imread(img_path)

            # Pengecekan apakah gambar tidak kosong
            if image is not None:
                # Lakukan preprocessing gambar
                preprocessed_image = preprocess_image(image)

                # Ekstrak teks dari gambar menggunakan Tesseract OCR
                text = extract_text(preprocessed_image)

                # Pengecekan apakah teks tidak kosong
                if text.strip():
                    # Resize gambar menjadi ukuran yang seragam (misalnya: (100, 100))
                    resized_image = cv2.resize(preprocessed_image, (1150, 306))  # Ganti ukuran sesuai kebutuhan

                    # Simpan gambar dan label ke list
                    images.append(resized_image)
                    labels.append(text)

# Konversi list gambar dan label menjadi numpy arrays
images_array = np.array(images)
labels_array = np.array(labels)

# Split dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(images_array, labels_array, test_size=0.2, random_state=42)
# Inisialisasi model (gunakan model yang sesuai dengan kebutuhan Anda, contohnya SVM)
model = SVC(C=1.0, kernel='rbf', gamma='auto')

# Melatih model menggunakan data latih
model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

# Prediksi menggunakan data uji
y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))

# Evaluasi performa model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Simpan model untuk digunakan nanti
model_filename = "number_plate_recognition_model.joblib"
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")