import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
np.random.seed(42)
plt.rcParams['figure.figsize'] = (9, 5)
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 11

print("1. Загрузка данных")
dataset = load_dataset("gvlassis/california_housing")
df = dataset['train'].to_pandas()
print("Форма датасета:", df.shape)
print("Колонки:", list(df.columns))

print("\n2. Первые 5 образцов:")
sample = df.head(5).copy()
sample.index = range(1, 6)

print("\n3. Разбивка на выборки")
feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                 'Population', 'AveOccup', 'Latitude', 'Longitude']
target_name = 'MedHouseVal'

X = df[feature_names].values
y = df[target_name].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape} ({len(X_train)/len(X)*100:.0f}%)")
print(f"Test: {X_test.shape} ({len(X_test)/len(X)*100:.0f}%)")
print("\n4. Масштабирование признаков...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("\n5. Инициализация весов и смещения")
n_samples, n_features = X_train.shape
print(f"Количество примеров: {n_samples}, Количество признаков: {n_features}")

w = np.random.randn(n_features)
b = np.random.randn()
print(f"Начальные веса (w): {w.round(3)}")
print(f"Начальное смещение (b): {b:.3f}")
print("\n6. Обучение")

lr = 0.01
epochs = 100
loss_history = []

for epoch in range(epochs):
    y_pred = X_train @ w + b
    loss = np.mean((y_train - y_pred) ** 2)
    loss_history.append(loss)
    error = y_train - y_pred
    dw = -2 / n_samples * (X_train.T @ error)
    db = -2 / n_samples * np.sum(error)
    w -= lr * dw
    b -= lr * db
    if (epoch + 1) % 5 == 0:
        print(f"Эпоха {epoch+1:3d}/{epochs} | Loss (MSE): {loss:.4f}")

plt.figure(figsize=(9, 5))
plt.plot(loss_history, color='blue', linewidth=2)
plt.xlabel("Эпоха")
plt.ylabel("MSE")
plt.title("Ручной GD: кривая обучения естественной тупости\n (Количество знаний от времени, 100 эпоха соответсвует 6 курсу)")
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.show()

print("\n7. Оценка на тестовой выборке")
y_pred_test = X_test @ w + b

mse = mean_squared_error(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2:  {r2:.4f}")

print("\n8. Сравнение первых 10 предсказаний с реальными ценами:")
comparison = pd.DataFrame({
    'Реальная цена': y_test[:10].round(3),
    'Предсказание': y_pred_test[:10].round(3),
    'Ошибка': (y_pred_test[:10] - y_test[:10]).round(3)
})
print(comparison)
