from collections.abc import Iterable

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler

import data


def main(use_location=True):
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
        mobility,
        rates,
        day_zeros,
        input_window=21,
        target_col="Doubling Rate",
        use_location=use_location,
    )
    model, score, split_ind = apply_regression(x, y)
    print(f"Doubling Rate Test Score: {score}")

    regression_coeff_plot(
        model,
        columns=list(mobility.columns)[2:],
        lags=21,
        sub_line=f"Target = Today's COVID Case Doubling Rate\nR^2 = {score:0.4f}",
        tag="doubling_rate",
        figsize=(8, 4),
        locations=sorted(set(state)),
    )

    x, y, state = prepare_regression_data(
        mobility,
        covid,
        day_zeros,
        input_window=14,
        target_col="positiveIncrease",
        use_location=use_location,
    )
    model, score, split_ind = apply_regression(x, y)
    print(f"New Cases Test Score: {score}")

    regression_coeff_plot(
        model,
        columns=list(mobility.columns)[2:],
        lags=14,
        sub_line=f"Target = Today's New Covid Cases\nR^2 = {score:0.4f}",
        tag="new_cases",
        locations=sorted(set(state)),
    )

    x, y, state = prepare_regression_data(
        mobility,
        covid,
        day_zeros,
        input_window=21,
        target_col="deathIncrease",
        use_location=use_location,
    )
    model, score, split_ind = apply_regression(x, y)
    print(f"New Deaths Test Score: {score}")

    regression_coeff_plot(
        model,
        columns=list(mobility.columns)[2:],
        lags=21,
        sub_line=f"Target = Today's COVID Deaths\nR^2 = {score:0.4f}",
        tag="new_deaths",
        locations=sorted(set(state)),
    )


def prepare_regression_data(
    feature_df,
    target_df,
    start_dates,
    input_window=7,
    target_col="positive",
    use_location=False,
):
    xs, ys, states = [], [], []
    for region in sorted(set(feature_df.Region) & set(target_df.Region)):
        xs.append([]), ys.append([]), states.append([])
        start_date = start_dates[region]
        state_mobility = feature_df[
            (feature_df.Region == region) & (feature_df.Date >= start_date)
        ].set_index("Date")
        state_mobility.drop(columns="Region", inplace=True)
        state_covid = target_df[target_df.Region == region]

        for i in range(len(state_mobility) - input_window):
            states[-1].append(region)
            xs[-1].append(state_mobility.iloc[i : i + input_window].values.flatten())
            ys[-1].append(state_covid[target_col].iloc[i + input_window])

    # Interleave xs, ys, and states to allow better handling of TS CV
    x, y, state = [], [], []
    for i in range(max(len(e) for e in ys)):
        for x_, y_, state_ in zip(xs, ys, states):
            if len(x_) > i:
                x.append(x_[i]), y.append(y_[i]), state.append(state_[i])

    x = np.array(x)
    if use_location:
        enc_state = LabelBinarizer().fit_transform(state)
        x = np.concatenate([x, enc_state], axis=-1)
    return x, np.array(y), state


def apply_regression(x, y, test_size=0.3, test_size_band=0.15):
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
        StandardScaler(), ElasticNetCV(max_iter=100_000, cv=TimeSeriesSplit()),
    )
    pipeline.fit(x_train, y_train)
    return pipeline, pipeline.score(x_test, y_test), split_ind


def regression_coeff_plot(
    model, columns, lags, sub_line=None, tag=None, figsize=(8, 4), locations=None,
):
    if not isinstance(lags, Iterable):
        lags = list(range(-lags, 0, 1))

    coeffs = (
        model[-1].coef_[: len(columns) * len(lags)].reshape((len(lags), len(columns)))
    )
    fig, ax = plt.subplots(figsize=figsize)
    # Reorient the array for a nicer display
    plt.imshow(
        coeffs.T, cmap="seismic", norm=MidpointNormalize(coeffs.min(), coeffs.max(), 0)
    )
    plt.ylabel("Feature")
    plt.yticks(np.arange(len(columns)), columns)
    plt.xlabel("Time Lag (Days)")
    plt.xticks(np.arange(len(lags)), lags)

    if sub_line:
        sub_line = f"\n{sub_line}"
    plt.title(f"Regression Coefficients{sub_line}")
    plt.colorbar(orientation="horizontal")
    plt.tight_layout()

    if tag:
        tag = f"_{tag}"
    plt.savefig(f"../results/regression_coeffs{tag}.png")
    plt.close(fig)

    if locations:
        coeffs = model[-1].coef_[len(columns) * len(lags) :]
        coeffs, locations = zip(
            *sorted(zip(coeffs, locations), key=lambda x: abs(x[0]), reverse=True)
        )
        coeffs, locations = np.array(coeffs), np.array(locations)
        inds = np.arange(len(coeffs))
        fig, ax = plt.subplots(figsize=(12, 5))
        plt.bar(
            inds,
            coeffs,
            color=plt.cm.get_cmap("seismic")(
                MidpointNormalize(coeffs.min(), coeffs.max(), 0)(coeffs)
            ),
        )
        plt.ylabel("Coefficient")
        plt.xlabel("State")
        plt.xticks(inds, locations, rotation=45, ha="right")
        plt.title(f"Regression Coefficients{sub_line}")
        plt.tight_layout()
        plt.savefig(f"../results/regression_coeffs{tag}_regions.png")
        plt.close(fig)


class MidpointNormalize(colors.Normalize):
    """
    Reference:
        https://stackoverflow.com/questions/25500541/matplotlib-bwr-colormap-always-centered-on-zero
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


if __name__ == "__main__":
    main()
