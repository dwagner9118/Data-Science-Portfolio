import pandas as pd
import matplotlib.pyplot as plt


def load_data(file_path):
    """Load USCIS case processing dataset."""
    return pd.read_csv(file_path)


def preprocess_data(df):
    """Basic cleaning for case analytics."""
    df = df.dropna()

    numeric_columns = ["completions", "workhours", "rate", "BenchMark"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_columns)

    return df


def analyze_workload(df):
    """Summarize workload patterns by service center."""
    summary = (
        df.groupby("service_center")[["completions", "workhours"]]
        .mean()
        .reset_index()
    )

    print("\nAverage workload by service center:")
    print(summary)

    return summary


def analyze_forms(df):
    """Summarize completions by form number."""
    form_summary = (
        df.groupby("form_number")["completions"]
        .sum()
        .reset_index()
        .sort_values(by="completions", ascending=False)
    )

    print("\nTop forms by completions:")
    print(form_summary.head(10))

    return form_summary


def plot_service_center_workload(summary):
    """Plot average completions by service center."""
    plt.figure()
    plt.bar(summary["service_center"], summary["completions"])
    plt.xlabel("Service Center")
    plt.ylabel("Average Completions")
    plt.title("Average Completions by Service Center")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_top_forms(form_summary):
    """Plot top 10 forms by completions."""
    top_forms = form_summary.head(10)

    plt.figure()
    plt.bar(top_forms["form_number"].astype(str), top_forms["completions"])
    plt.xlabel("Form Number")
    plt.ylabel("Total Completions")
    plt.title("Top 10 Forms by Completions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    df = load_data("uscis_case_data.csv")
    df_clean = preprocess_data(df)

    service_center_summary = analyze_workload(df_clean)
    form_summary = analyze_forms(df_clean)

    plot_service_center_workload(service_center_summary)
    plot_top_forms(form_summary)

    print("USCIS analytics completed successfully.")


if __name__ == "__main__":
    main()
