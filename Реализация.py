# ============================================
# Разработка полносвязной нейронной сети
# Анализ влияния климатических факторов # на фенологию растений
# ============================================
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# ============== ПАРАМЕТРИЗАЦИЯ ==============
# Параметры обучения
EPOCHS = 200
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42
# Параметры архитектуры сети
INPUT_FEATURES = 10
HIDDEN_LAYER_1 = 64
HIDDEN_LAYER_2 = 32
HIDDEN_LAYER_3 = 16
OUTPUT_CLASSES = 4
# Параметры оптимизатора
LEARNING_RATE = 0.001
# ============== ГЕНЕРАЦИЯ ДАТАСЕТА ==============
np.random.seed(RANDOM_STATE)
n_samples = 3000
# Генерируем климатические факторы
temperature = np.random.uniform(-20, 40, n_samples)
precipitation = np.random.uniform(0, 200, n_samples)
humidity = np.random.uniform(20, 100, n_samples)
solar_radiation = np.random.uniform(5, 30, n_samples)
day_length = np.random.uniform(6, 18, n_samples)
humidity_change = np.random.uniform(-30, 30, n_samples)
max_temp = np.random.uniform(-10, 50, n_samples)
min_temp = np.random.uniform(-30, 30, n_samples)
cloud_cover = np.random.uniform(0, 100, n_samples)
pressure = np.random.uniform(900, 1050, n_samples)
# Логика определения фенофазы
labels = []
for i in range(n_samples):
  if temperature[i] < 0 or day_length[i] < 10:
    labels.append(0) # Dormancy
  elif 5 <= temperature[i] < 15 and 10 <= day_length[i] < 14:
    labels.append(1) # Bud Break
  elif 15 <= temperature[i] < 25 and humidity[i] >= 50:
    labels.append(2) # Flowering
  else:
    labels.append(3) # Fruiting
labels = np.array(labels)
# Создаем DataFrame
X = np.column_stack([ temperature, precipitation, humidity, solar_radiation, day_length, humidity_change, max_temp, min_temp, cloud_cover, pressure ])
y = tf.keras.utils.to_categorical(labels, num_classes=OUTPUT_CLASSES)
print(f"Форма X: {X.shape}")
print(f"Форма y: {y.shape}")
print(f"Распределение классов: {np.unique(labels, return_counts=True)}")
# ============== ПРЕДОБРАБОТКА ДАННЫХ ==============
# Разделяем на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels )
# Нормализуем данные
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(f"Размер тренировочного набора: {X_train.shape}")
print(f"Размер тестового набора: {X_test.shape}")
# ============== ПОСТРОЕНИЕ МОДЕЛИ ==============
model = tf.keras.Sequential([ tf.keras.layers.Dense( HIDDEN_LAYER_1, activation='relu', input_shape=(INPUT_FEATURES,), name='input_layer' ), tf.keras.layers.Dropout(0.3), tf.keras.layers.Dense( HIDDEN_LAYER_2, activation='relu', name='hidden_1' ), tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense( HIDDEN_LAYER_3, activation='relu', name='hidden_2' ), tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense( OUTPUT_CLASSES, activation='softmax', name='output_layer' ) ])
# Компилируем модель
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile( optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'] )
model.summary()
# ============== ОБУЧЕНИЕ МОДЕЛИ ==============
history = model.fit( X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, verbose=1 )
# ============== ОЦЕНКА МОДЕЛИ ==============
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'\nТочность на тестовых данных: {test_accuracy*100:.2f}%')
# Предсказания на тестовом наборе
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
# ============== ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ==============
plt.figure(figsize=(15, 10))
# График точности
plt.subplot(2, 3, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Точность обучения')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()
plt.grid()
# График потерь
plt.subplot(2, 3, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Функция потерь')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend()
plt.grid()
# Матрица ошибок
plt.subplot(2, 3, 3)
cm = confusion_matrix(y_test_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Матрица ошибок')
plt.ylabel('True')
plt.xlabel('Predicted')
# Отчет о классификации
plt.subplot(2, 3, 4)
plt.axis('off')
report = classification_report( y_test_classes, y_pred_classes, target_names=['Dormancy', 'Bud Break', 'Flowering', 'Fruiting'], output_dict=True )
report_text = f"Precision: {report['weighted avg']['precision']:.3f}\n" + f"Recall: {report['weighted avg']['recall']:.3f}\n" + f"F1-Score: {report['weighted avg']['f1-score']:.3f}"
plt.text(0.1, 0.5, report_text, fontsize=12, family='monospace')
plt.tight_layout()
plt.show()
# ============== ТЕСТИРОВАНИЕ НА ПРИМЕРАХ ==============
phenology_phases = ['Dormancy', 'Bud Break', 'Flowering', 'Fruiting']
# Пример 1: Зимний покой
test_sample_1 = np.array([[[-15, 20, 40, 8, 8, -10, -5, -20, 60, 1020]]])
test_sample_1 = scaler.transform(test_sample_1.reshape(1, -1))
pred_1 = model.predict(test_sample_1)
print(f"\nПример 1 (Зима): {phenology_phases[np.argmax(pred_1[0])]} (уверенность: {np.max(pred_1[0])*100:.1f}%)")
# Пример 2: Весеннее распускание
test_sample_2 = np.array([[[10, 60, 65, 16, 12, 5, 18, 2, 30, 1013]]])
test_sample_2 = scaler.transform(test_sample_2.reshape(1, -1))
pred_2 = model.predict(test_sample_2)
print(f"Пример 2 (Весна): {phenology_phases[np.argmax(pred_2[0])]} (уверенность: {np.max(pred_2[0])*100:.1f}%)")
# Пример 3: Летнее цветение
test_sample_3 = np.array([[[22, 70, 80, 25, 15, 3, 28, 16, 20, 1010]]])
test_sample_3 = scaler.transform(test_sample_3.reshape(1, -1))
pred_3 = model.predict(test_sample_3)
print(f"Пример 3 (Лето): {phenology_phases[np.argmax(pred_3[0])]} (уверенность: {np.max(pred_3[0])*100:.1f}%)")
print("\n✅ Обучение и тестирование завершены!")

