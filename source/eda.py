import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import data


def main():
    df = data.load_covid_case_data(mode="states")

    peak_times = data.start_to_peak_time(df, indicator_col="positiveIncrease")
    states, peak_times = [
        np.array(x)
        for x in zip(*sorted(peak_times.items(), key=lambda x: x[1], reverse=True))
    ]
    peak_times = pd.DataFrame({"State": states, "Days to Case Growth Peak": peak_times})
    fig, ax = plt.subplots(figsize=(5, 8))
    sns.barplot(x="Days to Case Growth Peak", y="State", data=peak_times)
    plt.tight_layout()
    plt.savefig("../results/state_case_growth_peak.png")
    plt.close(fig)

    doubling_rates = data.get_covid_doubling_rates(df).stack().reset_index()
    doubling_rates.columns = [
        "Days Since First Case",
        "Region",
        "Days till Cases Double",
    ]
    fig, ax = plt.subplots()
    sns.lineplot(
        x="Days Since First Case",
        y="Days till Cases Double",
        hue="Region",
        data=doubling_rates,
        ax=ax,
        legend=False,
    )
    plt.title("Case Doubling Rates by State")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("../results/state_case_doubling.png")
    plt.close(fig)

    double_bars = doubling_rates.groupby(["Region"]).mean().reset_index()
    double_bars.sort_values("Days till Cases Double", ascending=True, inplace=True)
    fig, ax = plt.subplots(figsize=(5, 8))
    sns.barplot(
        x="Days till Cases Double", y="Region", data=double_bars, ax=ax,
    )
    plt.title("Case Doubling Rates by State")
    plt.ylabel("State")
    plt.tight_layout()
    plt.savefig("../results/state_case_doubling_bars.png")
    plt.close(fig)

    doubling_rates = (
        data.get_covid_doubling_rates(df, indicator_col="death").stack().reset_index()
    )
    doubling_rates.columns = [
        "Days Since First Death",
        "Region",
        "Days till Deaths Double",
    ]
    fig, ax = plt.subplots()
    sns.lineplot(
        x="Days Since First Death",
        y="Days till Deaths Double",
        hue="Region",
        data=doubling_rates,
        ax=ax,
        legend=False,
    )
    plt.title("Death Doubling Rates by State")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("../results/state_death_doubling.png")
    plt.close(fig)

    double_bars = doubling_rates.groupby(["Region"]).mean().reset_index()
    double_bars.sort_values("Days till Deaths Double", ascending=True, inplace=True)
    fig, ax = plt.subplots(figsize=(5, 8))
    sns.barplot(
        x="Days till Deaths Double", y="Region", data=double_bars, ax=ax,
    )
    plt.ylabel("State")
    plt.tight_layout()
    plt.savefig("../results/state_death_doubling_bars.png")
    plt.close(fig)

    us_df = data.load_google_covid19_mobility(mode="us")
    fig, ax = plt.subplots()
    sns.lineplot(
        x="Date", y="Relative Change (%)", hue="Mobility Type", data=us_df, ax=ax,
    )
    plt.title("Aggregated US Mobility Patterns")
    plt.xticks(rotation=45, ha="right", va="top")
    plt.ylim(-100, 100)
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig("../results/us.png")
    plt.close(fig)

    states_df = data.load_google_covid19_mobility(mode="states")
    fig, ax = plt.subplots()
    sns.lineplot(
        x="Date", y="Relative Change (%)", hue="Mobility Type", data=states_df, ax=ax,
    )
    plt.title("Aggregated US State Mobility Patterns")
    plt.xticks(rotation=45, ha="right", va="top")
    plt.ylim(-100, 100)
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig("../results/states.png")
    plt.close(fig)

    us_case_df = data.load_covid_case_data(mode="us")
    us_df = us_df.pivot(
        index="Date", columns="Mobility Type", values="Relative Change (%)"
    )
    us_combined = pd.merge(us_df, us_case_df, on=["Date"], how="left")
    us_combined.set_index("Date", inplace=True)

    corr, labels = correlation_matrix(
        us_combined.fillna(value=0),
        group1=[
            "Grocery And Pharmacy",
            "Parks",
            "Residential",
            "Retail And Recreation",
            "Transit Stations",
            "Workplaces",
        ],
        group2=[
            "pending",
            "negative",
            "positive",
            "hospitalizedCumulative",
            "inIcuCumulative",
            "onVentilatorCumulative",
            "death",
            "recovered",
        ],
    )
    plot_correlation_matrix(
        corr,
        tick_labels=labels,
        title="US (Aggregated) Mobility-Covid Correlations",
        tag="us_pearson",
    )

    state_case_df = data.load_covid_case_data(mode="states")
    states_combined = pd.merge(
        states_df, state_case_df, on=["Date", "Region"], how="left"
    )


def correlation_matrix(
    df, group1=None, group2=None, ignore=None, correlation_func=stats.spearmanr
):
    columns = sorted(df.columns)
    if ignore:
        columns = sorted(set(columns) - set(ignore))

    if not group1:
        group1 = columns
    if not group2:
        group2 = columns

    corr_mat = np.zeros((len(group1), len(group2)))
    for i, col1 in enumerate(group1):
        for j, col2 in enumerate(group2):
            corr, *_ = correlation_func(df[col1].values, df[col2].values)
            corr_mat[i, j] = corr
    return corr_mat, [group1, group2]


def plot_correlation_matrix(
    corr_mat, tick_labels=None, axis_labels=None, title=None, cmap="viridis", tag=""
):
    fig, ax = plt.subplots()
    plt.imshow(corr_mat, cmap=cmap, vmax=1, vmin=-1)
    if tick_labels:
        y_labels, x_labels = tick_labels
        y_labels = [x.strip().replace("_", " ").title() for x in y_labels]
        x_labels = [x.strip().replace("_", " ").title() for x in x_labels]
        plt.yticks(np.arange(len(tick_labels[0])), y_labels)
        plt.xticks(np.arange(len(tick_labels[1])), x_labels, rotation=45, ha="right")

    if axis_labels:
        plt.ylabel(axis_labels[0])
        plt.xlabel(axis_labels[1])

    if tag:
        plt.title(tag.replace("_", " ").title())
        tag = f"_{tag}"

    if title:
        plt.title(title)

    plt.colorbar()
    plt.tight_layout()

    plt.savefig(f"../results/correlation{tag}.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
