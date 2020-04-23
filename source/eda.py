import matplotlib.pyplot as plt
import seaborn as sns

import data


def main():
    df = data.load_google_covid19_mobility()

    us_df = data.clean_google_covid19_mobility(df, mode="us")
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

    us_states_df = data.clean_google_covid19_mobility(df, mode="us_states")
    fig, ax = plt.subplots()
    sns.lineplot(
        x="Date",
        y="Relative Change (%)",
        hue="Mobility Type",
        data=us_states_df,
        ax=ax,
    )
    plt.title("Aggregated US State Mobility Patterns")
    plt.xticks(rotation=45, ha="right", va="top")
    plt.ylim(-100, 100)
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig("../results/states.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
