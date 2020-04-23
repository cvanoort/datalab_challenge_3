from pathlib import Path

import pandas as pd


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
):
    if not path.is_file() or stale_file(path, update_window=update_window):
        df = download_google_covid19_mobility()
    else:
        df = pd.read_csv(path, index_col=0)
    df.date = pd.to_datetime(df.date)
    return df


def clean_google_covid19_mobility(df, mode="us"):
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
    elif mode == "us_states":
        us_states_ind = (
            (df.country_region_code == "US")
            & ~df.sub_region_1.isna()
            & df.sub_region_2.isna()
        )
        us_states_df = (
            df[us_states_ind]
            .dropna(axis=1)
            .drop(["country_region_code", "country_region"], axis=1)
        )
        # Clean up the column labels for a better legend later
        us_states_df.rename(columns=lambda x: x.replace("_", " ").title(), inplace=True)
        us_states_df = (
            us_states_df.set_index(["Date", "Sub Region 1"]).stack().reset_index()
        )
        us_states_df.columns = [
            "Date",
            "Region",
            "Mobility Type",
            "Relative Change (%)",
        ]
        return us_states_df
    elif mode == "us_counties":
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
    mode="us", update_window=pd.Timedelta(days=1),
):
    assert mode in {"us", "states"}
    path = Path(f"../data/covid_cases_{mode}.csv")

    if not path.is_file() or stale_file(path, update_window=update_window):
        df = download_covid_case_data(mode=mode)
    else:
        df = pd.read_csv(path, index_col=0)
    df.date = pd.to_datetime(df.date)
    return df


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
