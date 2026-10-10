import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.regularizers import l2

np.random.seed(42)
tf.random.set_seed(42)

plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 11

print("Загрузка датасета")
dataset = load_dataset("stanfordnlp/imdb")
print(dataset)

print("\nРазбивка на train/test")
train_texts = dataset['train']['text']
train_labels = dataset['train']['label']
test_texts  = dataset['test']['text']
test_labels = dataset['test']['label']

y_train = np.array(train_labels)
y_test  = np.array(test_labels)

print(f"Обучающих отзывов: {len(train_texts)}")
print(f"Тестовых отзывов:  {len(test_texts)}")
print(f"Форма y_train: {y_train.shape}, y_test: {y_test.shape}")
print(f"Распределение меток в train (0, 1): {np.bincount(y_train)}")

print("\nТокенизация")
vocab_size = 10000
max_len = 200

tokenizer = Tokenizer(num_words=vocab_size, oov_token="")
tokenizer.fit_on_texts(train_texts)

print(f"Уникальных слов в train: {len(tokenizer.word_index)}")
print(f"Оставляем {vocab_size} самых частых")
print("\nПаддинг")
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences  = tokenizer.texts_to_sequences(test_texts)

X_train = pad_sequences(train_sequences, maxlen=max_len,
                        padding='post', truncating='post')
X_test  = pad_sequences(test_sequences, maxlen=max_len,
                        padding='post', truncating='post')

print(f"Форма X_train: {X_train.shape}")
print(f"Форма X_test:  {X_test.shape}")

#model = Sequential([
#    Embedding(input_dim=vocab_size, output_dim=16, input_length=max_len),
#    GlobalAveragePooling1D(),
#    Dense(16, activation='relu'),
#    Dense(1, activation='sigmoid')
#])
#предыдущая просто для истории, на ней очень сильное переобучение было
#что касается графиков, величины дропаутов скорее оптимальные,
#по крайней мере, реже переобучение появляется, чем при других
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=8, input_length=max_len),
    GlobalAveragePooling1D(),
    Dropout(0.3),
    Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nОбучение( в узких кругах также известное, как neuron activation) )")
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'],     label='train loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='val loss',   linewidth=2)
axes[0].set_xlabel("Эпоха")
axes[0].set_ylabel("Binary Crossentropy")
axes[0].set_title("Функция потерь (train vs val)")
axes[0].legend()

axes[1].plot(history.history['accuracy'],     label='train accuracy', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='val accuracy',   linewidth=2)
axes[1].set_xlabel("Эпоха")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Точность (train vs val)")
axes[1].legend()

plt.tight_layout()
plt.show()

print("\nОценка на тестовой выборке")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss:     {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

print("\nПредсказание для 5 образцов")
n_show = 5
X_sample = X_test[:n_show]
y_true   = y_test[:n_show]

y_proba = model.predict(X_sample, verbose=0).flatten()
y_pred  = (y_proba > 0.5).astype(int)
abs_err = np.abs(y_true - y_proba)

result = pd.DataFrame({
    'Текст (первые 70 символов)': [t[:70] + "..." for t in test_texts[:n_show]],
    'Реальная метка':             y_true,
    'Вероятность (sigmoid)':      y_proba.round(4),
    'Предсказание':               y_pred,
    'Ошибка |y - ŷ|':             abs_err.round(4),
    'Правильно?':                 (y_pred == y_true)
})
print(result.to_string(index=True))
print(f"\nСредняя ошибка |y - ŷ| по {n_show} образцам: {abs_err.mean():.4f}")

print("\nМетрики на всей тестовой выборке...")
y_pred_proba_all = model.predict(X_test, verbose=0).flatten()
y_pred_all       = (y_pred_proba_all > 0.5).astype(int)

acc  = accuracy_score(y_test, y_pred_all)
prec = precision_score(y_test, y_pred_all)
rec  = recall_score(y_test, y_pred_all)
f1   = f1_score(y_test, y_pred_all)

print(f"Accuracy  (точность):    {acc:.4f} ({acc * 100:.2f}%)")
print(f"Precision (точность+):   {prec:.4f}")
print(f"Recall    (полнота):     {rec:.4f}")
print(f"F1-score  (гармоническое): {f1:.4f}")

print("\nПроверка на своих примерах, ну как своих, естественная тупость отказывается работать, поэтому частично делегируем")

def predict_sentiment(text, model, tokenizer, max_len=200):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len,
                           padding='post', truncating='post')
    proba = model.predict(padded, verbose=0)[0][0]
    label = "ПОЛОЖИТЕЛЬНЫЙ" if proba > 0.5 else "ОТРИЦАТЕЛЬНЫЙ"
    return label, proba

my_reviews = [
    #позитивные
    "This movie was absolutely fantastic! I loved every minute of it.",
    "A masterpiece of cinema. Brilliant storytelling and acting.",
    "One of the best films I've ever seen. The cinematography is stunning.",
    "An emotional rollercoaster that left me speechless. Highly recommended!",
    "The performances were phenomenal and the script was razor-sharp.",
    #негативные
    "Terrible film, complete waste of time. The acting was awful.",
    "Boring, predictable, and utterly forgettable. Do not watch.",
    "I fell asleep halfway through. The plot made no sense at all.",
    "A complete disaster from start to finish. Avoid at all costs.",
    "The dialogue was cringeworthy and the ending was a total cop-out.",
    #сомнительно, но окей
    "It was okay, not great but not bad either.",
    "Some parts were interesting, but overall it felt too long.",
    "Decent acting, weak script. An average film at best.",
    "Not my cup of tea, but I can see why some people like it.",
    #вопрос с подвохом
    "Oh great, another sequel nobody asked for. Truly groundbreaking.",
    "Two hours of my life I'll never get back. Thanks, Hollywood.",
    "It's so bad it's almost good. Almost.",
    #а тут, думаю, узнаешь откуда фразы)
    "And no matter what happens to you, don’t take anything to heart. Few things in life stay important for long.",
    "If you want to do something, never ask about the consequences. Otherwise, you’ll never do anything.",
    "What can one person give to another except a drop of warmth? And what could be more than that?",
    "Only the one who has lost everything worth living for is free.",
    "Repentance is the most useless thing in the world. You can’t bring anything back. Nothing can be fixed. Otherwise, we would all be saints."
]

for review in my_reviews:
    label, proba = predict_sentiment(review, model, tokenizer)
    print(f"Отзыв: {review[:70]}...")
    print(f"  → {label} (вероятность: {proba:.4f})\n")

input()
