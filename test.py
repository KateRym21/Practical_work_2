import pandas as pd  # Імпортуємо бібліотеку pandas для роботи з таблицями
import numpy as np  # Імпортуємо бібліотеку numpy для числових операцій
from sklearn.ensemble import RandomForestRegressor  # Імпортуємо модель випадкового лісу для регресії
from sklearn.model_selection import train_test_split  # Функція для поділу даних на навчальні та тестові
from sklearn.preprocessing import StandardScaler  # Масштабування даних
import joblib  # Для збереження та завантаження моделі

# Завантаження даних з CSV-файлу
data = pd.read_csv("train.csv")


def preprocess_data(data):
    """Обробка пропущених значень та категоріальних змінних"""
    for column in data.columns:
        if data[column].dtype == "object":  # Якщо змінна текстова (категоріальна)
            data[column].fillna("Missing", inplace=True)  # Заповнюємо пропущені значення рядком "Missing"
        else:
            data[column].fillna(data[column].median(), inplace=True)  # Числові заповнюємо медіаною

    # Перетворення категоріальних змінних у числовий формат (one-hot encoding)
    data = pd.get_dummies(data, drop_first=True)

    return data  # Повертаємо оброблені дані


# Обробка вхідних даних
processed_data = preprocess_data(data)

# Відокремлюємо ознаки (фактори) та цільову змінну (ціна будинку)
X = processed_data.drop("SalePrice", axis=1)  # Видаляємо "SalePrice" – це наша цільова змінна
y = processed_data["SalePrice"]  # Виділяємо ціну будинку як цільову змінну

# Збережемо назви стовпців перед масштабуванням
X_columns = X.columns

# Масштабування ознак
scaler = StandardScaler()  # Ініціалізуємо масштабатор
X = scaler.fit_transform(X)  # Масштабуємо дані

# Поділ на навчальну та тестову вибірки (80% - навчальні, 20% - тестові)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ініціалізація та навчання моделі випадкового лісу
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)  # Навчання моделі на тренувальних даних

# Збереження моделі та масштабатора
joblib.dump(model, "house_price_model.pkl")  # Зберігаємо модель у файл
joblib.dump(scaler, "scaler.pkl")  # Зберігаємо масштабатор
joblib.dump(X_columns, "X_columns.pkl")  # Зберігаємо список стовпців


def predict_price(input_features):
    """Функція прогнозування ціни будинку"""
    model = joblib.load("house_price_model.pkl")  # Завантажуємо збережену модель
    scaler = joblib.load("scaler.pkl")  # Завантажуємо збережений масштабатор
    X_columns = joblib.load("X_columns.pkl")  # Завантажуємо список стовпців

    # Створюємо DataFrame з введених користувачем значень
    input_df = pd.DataFrame([input_features])

    # Попередня обробка введених даних
    input_df = preprocess_data(input_df)

    # Додаємо відсутні стовпці (якщо є)
    missing_cols = set(X_columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0  # Додаємо відсутні стовпці і заповнюємо їх нулями

    # Переконуємося, що порядок стовпців збігається з початковими даними
    input_df = input_df[X_columns]

    # Масштабуємо вхідні дані
    input_scaled = scaler.transform(input_df)

    # Прогнозуємо ціну
    predicted_price = model.predict(input_scaled)
    return predicted_price[0]  # Повертаємо передбачену ціну


# Введення характеристик будинку користувачем
print("Введіть характеристики будинку:")

features = {
    "MSSubClass": int(input("MSSubClass (клас будівлі): ")),  # Категорія будівлі
    "LotArea": float(input("LotArea (площа ділянки, кв. футів): ")),  # Площа земельної ділянки
    "OverallQual": int(input("OverallQual (загальна якість будинку): ")),  # Загальна якість будівлі
    "YearBuilt": int(input("YearBuilt (рік побудови): ")),  # Рік побудови будинку
    "GrLivArea": float(input("GrLivArea (житлова площа, кв. футів): ")),  # Житлова площа будинку
    "FullBath": int(input("FullBath (кількість повних ванних кімнат): ")),  # Кількість повних ванних кімнат
    "BedroomAbvGr": int(input("BedroomAbvGr (кількість спалень вище підвалу): ")),  # Кількість спалень
    "TotRmsAbvGrd": int(input("TotRmsAbvGrd (загальна кількість кімнат): ")),  # Загальна кількість кімнат
}

# Прогноз ціни будинку
price = predict_price(features)
print(f"Прогнозована ціна будинку: ${price:,.2f}")  # Вивід прогнозованої ціни
