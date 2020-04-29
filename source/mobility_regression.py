import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler

import data


def main():
    mobility = (
        data.load_google_covid19_mobility(mode="states")
        .set_index(["Date", "Region", "Mobility Type"])
        .unstack()
        .reset_index()
    )
    mobility.columns = ["Date", "Region"] + list(mobility.columns.droplevel())[2:]

    covid = data.load_covid_case_data(mode="states")
    day_zeros = data.first_occurrence_date(covid)
    rates = data.get_covid_doubling_rates(df=covid).stack().reset_index()
    rates.columns = ["Date", "Region", "Doubling Rate"]

    x, y, state = prepare_regression_data(
        mobility, rates, day_zeros, target="Doubling Rate"
    )
    model, score, split_ind = apply_regression(x, y)
    print(f"Test Score: {score}")

    regression_coeff_plot(
        model,
        columns=list(mobility.columns)[2:],
        lags=[f"T-{i}" for i in range(7, 0, -1)],
        sub_line=f"Target = Case Doubling Rate\nR^2 = {score:0.4f}",
    )


def prepare_regression_data(
    mobility, covid, day_zero, input_window=7, target="positive", use_location=False
):
    x, y, state = [], [], []
    for region in sorted(set(mobility.Region) & set(covid.Region)):
        start_date = day_zero[region]
        state_mobility = mobility[
            (mobility.Region == region) & (mobility.Date >= start_date)
        ].set_index("Date")
        state_mobility.drop(columns="Region", inplace=True)
        state_covid = covid[covid.Region == region]

        for i in range(len(state_mobility) - input_window):
            state.append(region)
            x.append(state_mobility.iloc[i : i + input_window].values.flatten())
            y.append(state_covid[target].iloc[i + input_window])

    x = np.array(x)
    if use_location:
        bin_state = LabelBinarizer().fit_transform(state)
        x = np.concatenate([x, bin_state], axis=-1)
    return np.array(x), np.array(y), state


def apply_regression(x, y, test_size=0.2, test_size_band=0.1):
    low = int((1 - test_size - test_size_band) * len(x))
    high = int((1 - test_size + test_size_band) * len(x))
    split_ind = np.random.randint(low, high)
    x_train, x_test, y_train, y_test = (
        x[:split_ind],
        x[split_ind:],
        y[:split_ind],
        y[split_ind:],
    )
    pipeline = make_pipeline(
        # PolynomialFeatures(degree=2, interaction_only=True),
        StandardScaler(),
        ElasticNetCV(max_iter=100_000, cv=TimeSeriesSplit()),
    )
    pipeline.fit(x_train, y_train)
    return pipeline, pipeline.score(x_test, y_test), split_ind


def regression_coeff_plot(model, columns, lags, sub_line=None):
    coeffs = model[-1].coef_.reshape((len(lags), len(columns)))
    fig, ax = plt.subplots()
    # Reorient the array for a nicer display
    plt.imshow(coeffs.T, cmap="viridis")
    plt.ylabel("Feature")
    plt.yticks(np.arange(len(columns)), columns)
    plt.xlabel("Lag")
    plt.xticks(np.arange(len(lags)), lags)

    if sub_line:
        sub_line = f"\n{sub_line}"
    plt.title(f"Regression Coefficients{sub_line}")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("../results/regression_coeffs.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
