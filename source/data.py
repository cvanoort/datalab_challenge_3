from pathlib import Path
import numpy as np

import pandas as pd
import us


def download_google_covid19_mobility(
    path=Path("../data/google_covid19_mobility.csv"),
    url="https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv",
):
    df = pd.read_csv(url)
    df.to_csv(path)
    return df


def load_google_covid19_mobility(
    path=Path("../data/google_covid19_mobility.csv"),
    update_window=pd.Timedelta(days=1),
    clean=True,
    mode="us",
):
    assert mode in {"us", "states", "counties"}

    if not path.is_file() or stale_file(path, update_window=update_window):
        df = download_google_covid19_mobility()
    else:
        df = pd.read_csv(path, index_col=0)

    if clean:
        df = clean_google_covid19_mobility(df, mode=mode)
    return df


def clean_google_covid19_mobility(df, mode="us"):
    assert mode in {"us", "states", "counties"}

    df.date = pd.to_datetime(df.date)
    df.rename(
        columns=lambda x: x.replace("_percent_change_from_baseline", ""), inplace=True
    )

    if mode == "us":
        # Select the rows associated with the US (aggregated)
        us_ind = (
            (df.country_region_code == "US")
            & df.sub_region_1.isna()
            & df.sub_region_2.isna()
        )
        us_df = (
            df[us_ind]
            .dropna(axis=1)
            .drop(["country_region_code", "country_region"], axis=1)
        )

        # Clean up the column labels for a better legend later
        us_df.rename(columns=lambda x: x.replace("_", " ").title(), inplace=True)
        us_df = us_df.set_index("Date").stack().reset_index()
        us_df.columns = ["Date", "Mobility Type", "Relative Change (%)"]
        return us_df
    elif mode == "states":
        states_ind = (
            (df.country_region_code == "US")
            & ~df.sub_region_1.isna()
            & df.sub_region_2.isna()
        )
        states_df = (
            df[states_ind]
            .dropna(axis=1)
            .drop(["country_region_code", "country_region"], axis=1)
        )
        # Clean up the column labels for a better legend later
        states_df.rename(columns=lambda x: x.replace("_", " ").title(), inplace=True)
        states_df = states_df.set_index(["Date", "Sub Region 1"]).stack().reset_index()
        states_df.columns = [
            "Date",
            "Region",
            "Mobility Type",
            "Relative Change (%)",
        ]
        return states_df
    elif mode == "counties":
        # counties_ind = (df.country_region_code == "US") & ~df.sub_region_1.isna() & ~df.sub_region_2.isna()
        raise NotImplementedError()
    else:
        raise ValueError(f"Unexpected value for mode: {mode}.")


def download_covid_case_data(mode="us"):
    assert mode in {"us", "states"}
    url = f"https://covidtracking.com/api/v1/{mode}/daily.csv"
    path = Path(f"../data/covid_cases_{mode}.csv")
    df = pd.read_csv(url)
    df.to_csv(path)
    return df


def load_covid_case_data(
    mode="us", update_window=pd.Timedelta(days=1), clean=True,
):
    assert mode in {"us", "states"}
    path = Path(f"../data/covid_cases_{mode}.csv")

    if not path.is_file() or stale_file(path, update_window=update_window):
        df = download_covid_case_data(mode=mode)
    else:
        df = pd.read_csv(path, index_col=0)

    if clean:
        df = clean_covid_case_data(df, mode=mode)
    return df


def clean_covid_case_data(df, mode="us"):
    assert mode in {"us", "states"}
    df.drop(["hash", "fips"], axis=1, inplace=True)
    df.dropna(axis=1, inplace=True, how="all")
    df.date = pd.to_datetime(df.date, format="%Y%m%d")
    df.rename(columns={"date": "Date"}, inplace=True)
    if mode == "states":
        df.state = df.state.apply(
            lambda x: us.states.lookup(x).name if us.states.lookup(x) else x
        )
        df.state = df.state.apply(lambda x: "District of Columbia" if x == "DC" else x)
        # Drop the territories so we can focus on the states and DC.
        indicator = df.state.apply(lambda x: True if us.states.lookup(x) in us.STATES or (x == "District of Columbia") else False)
        df = df[indicator]
        df.rename(columns={"state": "Region"}, inplace=True)

    return df


def time_normalize(df, group_col="Region", indicator_col="positive"):
    # Align data to date of first positive COVID case
    chunks = dict()
    for label, chunk in df.groupby(group_col):
        chunk.set_index("Date", inplace=True)
        chunk.sort_index(inplace=True)
        chunk = chunk[chunk[indicator_col] > 0]
        chunk.index = pd.Series(np.arange(len(chunk)), name="Days Since First Case")
        if len(chunk):
            chunks[label] = chunk
        else:
            print(f"No positive cases in {label}")
    return chunks


def get_covid_doubling_rates(group_col="Region", indicator_col="positive"):
    state_case_df = load_covid_case_data(mode="states")
    chunks = time_normalize(
        state_case_df, group_col=group_col, indicator_col=indicator_col
    )

    doubling_rates = dict()
    for label, chunk in chunks.items():
        doubling_rates[label] = chunk[indicator_col] / (
            1 + chunk[indicator_col + "Increase"].clip(0).rolling(7).mean()
        )

    chunk_len = (
        (state_case_df[indicator_col] > 0)
        .groupby(state_case_df[group_col])
        .sum()
        .astype(int)
        .max()
    )
    drs = np.zeros((chunk_len, len(doubling_rates)), dtype=float)
    drs[:] = np.nan
    columns, values = zip(*doubling_rates.items())
    for i, vs in enumerate(values):
        drs[: len(vs), i] = vs

    doubling_rates = pd.DataFrame(drs, columns=columns)
    doubling_rates.index.name = "Days Since First Case"
    return doubling_rates


def stale_file(path, update_window=pd.Timedelta(days=1)):
    last_modified = pd.to_datetime(path.stat().st_mtime, unit="s")
    return pd.Timestamp.now() - last_modified > update_window


if __name__ == "__main__":
    import pandas_profiling

    pandas_profiling.ProfileReport(
        load_covid_case_data(mode="us"),
        title="Pandas Profiling Report",
        html={"style": {"full_width": True}},
    ).to_file("../results/data_report_us_cases.html")

    pandas_profiling.ProfileReport(
        load_covid_case_data(mode="states"),
        title="Pandas Profiling Report",
        html={"style": {"full_width": True}},
    ).to_file("../results/data_report_states_cases.html")

    pandas_profiling.ProfileReport(
        load_google_covid19_mobility(),
        title="Pandas Profiling Report",
        html={"style": {"full_width": True}},
    ).to_file("../results/data_report_google.html")
