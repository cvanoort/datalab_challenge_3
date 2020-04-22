import matplotlib.pyplot as plt
import seaborn as sns

import data


def main():
    df = data.load_google_covid19_mobility()
    us_ind = (df.country_region_code == "US") & df.sub_region_1.isna() & df.sub_region_2.isna()
    states_ind = (df.country_region_code == "US") & ~df.sub_region_1.isna() & df.sub_region_2.isna()
    # counties_ind = (df.country_region_code == "US") & ~df.sub_region_1.isna() & ~df.sub_region_2.isna()

    fig, ax = plt.subplots()
    sns.lineplot(
        x="date",
        y="retail_and_recreation_percent_change_from_baseline",
        data=df[us_ind],
        ax=ax,
    )
    plt.xticks(rotation=45, ha="right", va="top")
    plt.tight_layout()
    plt.savefig("../results/us.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.lineplot(
        x="date",
        y="retail_and_recreation_percent_change_from_baseline",
        hue="sub_region_1",
        data=df[states_ind],
        legend=False,
        ax=ax,
    )
    plt.xticks(rotation=45, ha="right", va="top")
    plt.tight_layout()
    plt.savefig("../results/states.png")
    plt.close(fig)

    # Too slow... figure out a better way to dig into this.
    # fig, ax = plt.subplots()
    # sns.lineplot(
    #     x="date",
    #     y="retail_and_recreation_percent_change_from_baseline",
    #     hue="sub_region_2",
    #     data=df[counties_ind],
    #     legend=False,
    #     ax=ax,
    # )
    # plt.xticks(rotation=45, ha="right", va="top")
    # plt.tight_layout()
    # plt.savefig("../results/counties.png")
    # plt.close("all")


if __name__ == '__main__':
    main()
