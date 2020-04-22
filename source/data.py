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
