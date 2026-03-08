import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def load_data(file_path):
    """Load Fitbit dataset."""
    return pd.read_csv(file_path)


def preprocess_data(df):
    """Basic cleaning and feature selection."""
    df = df.dropna()

    selected_columns = [
        "TotalSteps",
        "VeryActiveMinutes",
        "FairlyActiveMinutes",
        "LightlyActiveMinutes",
        "SedentaryMinutes",
        "Calories"
    ]

    return df[selected_columns]


def exploratory_analysis(df):
    """Create simple exploratory plots."""
    plt.figure()
    plt.scatter(df["TotalSteps"], df["Calories"])
    plt.xlabel("Total Steps")
    plt.ylabel("Calories")
    plt.title("Total Steps vs Calories")
    plt.show()

    plt.figure()
    df["Calories"].hist()
    plt.xlabel("Calories")
    plt.ylabel("Frequency")
    plt.title("Distribution of Calories Burned")
    plt.show()


def build_model(df):
    """Train regression model to predict calories."""
    X = df.drop("Calories", axis=1)
    y = df["Calories"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Mean Absolute Error:", round(mae, 2))
    print("R-squared:", round(r2, 4))

    return model, X_test, y_test, predictions


def plot_predictions(y_test, predictions):
    """Plot actual vs predicted calories."""
    plt.figure()
    plt.scatter(y_test, predictions)
    plt.xlabel("Actual Calories")
    plt.ylabel("Predicted Calories")
    plt.title("Actual vs Predicted Calories")
    plt.show()


def main():
    df = load_data("fitbit_data.csv")
    df_clean = preprocess_data(df)

    exploratory_analysis(df_clean)

    model, X_test, y_test, predictions = build_model(df_clean)

    plot_predictions(y_test, predictions)

    print("Fitbit analysis completed successfully.")


if __name__ == "__main__":
    main()
