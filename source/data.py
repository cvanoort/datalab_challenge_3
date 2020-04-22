from pathlib import Path

import pandas as pd
import us


def is_us_state(val):
    return True if us.states.lookup(val) else False


def load_github_covid19_mobility(
    path=Path("../data/google-covid19-mobility-reports/data/processed/mobility_reports.csv"),
    us_only=True,
):
    df = pd.read_csv(path)
    df.updated_at = pd.to_datetime(df.updated_at)

    if us_only:
        df = df[df.region.apply(is_us_state).values]
        df.reset_index(drop=True, inplace=True)

    return df


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
    last_modified = pd.to_datetime(path.stat().st_mtime, unit="s")
    if not path.is_file():
        df = download_google_covid19_mobility()
    elif pd.Timestamp.now() - last_modified > update_window:
        df = download_google_covid19_mobility()
    else:
        df = pd.read_csv(path, index_col=0)
    df.date = pd.to_datetime(df.date)
    return df


def clean_google_covid19_mobility(
        df,
        mode="us",
):
    df.rename(columns=lambda x: x.replace("_percent_change_from_baseline", ""), inplace=True)

    if mode == "us":
        # Select the rows associated with the US (aggregated)
        us_ind = (df.country_region_code == "US") & df.sub_region_1.isna() & df.sub_region_2.isna()
        us_df = df[us_ind].dropna(axis=1).drop(["country_region_code", "country_region"], axis=1)

        # Clean up the column labels for a better legend later
        us_df.rename(columns=lambda x: x.replace("_", " ").title(), inplace=True)
        us_df = us_df.set_index("Date").stack().reset_index()
        us_df.columns = ["Date", "Mobility Type", "Relative Change (%)"]
        return us_df
    elif mode == "us_states":
        us_states_ind = (df.country_region_code == "US") & ~df.sub_region_1.isna() & df.sub_region_2.isna()
        us_states_df = df[us_states_ind].dropna(axis=1).drop(["country_region_code", "country_region"], axis=1)
        # Clean up the column labels for a better legend later
        us_states_df.rename(columns=lambda x: x.replace("_", " ").title(), inplace=True)
        us_states_df = us_states_df.set_index(["Date", "Sub Region 1"]).stack().reset_index()
        us_states_df.columns = ["Date", "Region", "Mobility Type", "Relative Change (%)"]
        return us_states_df
    elif mode == "us_counties":
        # counties_ind = (df.country_region_code == "US") & ~df.sub_region_1.isna() & ~df.sub_region_2.isna()
        raise NotImplementedError()
    else:
        raise ValueError(f"Unexpected value for mode: {mode}.")


if __name__ == '__main__':
    # Prevent random tkinter errors:
    #     https://github.com/pandas-profiling/pandas-profiling/issues/373
    import matplotlib
    matplotlib.use('Agg')
    import pandas_profiling

    report = pandas_profiling.ProfileReport(
        load_github_covid19_mobility(us_only=False),
        title="Pandas Profiling Report",
        html={"style": {"full_width": True}},
    )
    report.to_file("../results/data_report_global.html")

    report = pandas_profiling.ProfileReport(
        load_github_covid19_mobility(us_only=True),
        title="Pandas Profiling Report",
        html={"style": {"full_width": True}},
    )
    report.to_file("../results/data_report_us.html")

    report = pandas_profiling.ProfileReport(
        load_google_covid19_mobility(),
        title="Pandas Profiling Report",
        html={"style": {"full_width": True}},
    )
    report.to_file("../results/data_report_google.html")
