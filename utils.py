import matplotlib.pyplot as plt
import numpy as np
import itertools
import seaborn as sns
import pandas as pd
plt.style.use("fivethirtyeight")


# Functions for univariate analysis
#----------------------------------------------------------------------------------

def make_barplot(dataframe, cat_vars):
    for cat_var in cat_vars:
        # Compute percentages
        percentages = dataframe[cat_var].value_counts(normalize=True) * 100

        # Size of the figure
        plt.figure(figsize=(16, 9))

        # Create the bar plot
        my_fig = percentages.plot(kind="bar")
        plt.ylabel("Percentage (%)")
        plt.xlabel(cat_var)

        # Rotate the x-axis labels
        plt.xticks(rotation=0)

        # Add the values as text on top of the bars
        for i, valor in enumerate(percentages):
            my_fig.text(i, valor/2, f"{valor:.1f}%", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=10)

        # Control the lines of the grid
        my_fig.grid(axis="x", visible=False)
        my_fig.grid(axis="y", visible=True)

        plt.tight_layout()
        plt.show()


def make_discrete_barplot(dataframe, feature, custom_ticks=None, unit=''):

    # Size of the figure
    plt.figure(figsize=(16, 9))

    # Count values
    value_counts = dict(dataframe[feature].value_counts())
    data = list(value_counts.values())
    labels = list(value_counts.keys())

    # Create bars
    plt.bar(labels, data, width=0.8)

    # Custom x-axis ticks
    if custom_ticks is not None:
        plt.xticks(custom_ticks)

    # Set axis labels
    plt.xlabel(feature)
    plt.ylabel('Frequency')

    # Control the lines of the grid
    plt.grid(axis="x", visible=False)
    plt.grid(axis="y", visible=True)

    # Set the ticks
    plt.tick_params(axis='x', pad=10)
    plt.tick_params(axis='y', pad=5)

    # Statistics
    mean_val = dataframe[feature].mean()
    std_val = dataframe[feature].std()
    median_val = dataframe[feature].median()
    min_val = dataframe[feature].min()
    max_val = dataframe[feature].max()
    skew_val = dataframe[feature].skew()

    stats_text = (
        f"Mean: {mean_val:.2f} {unit}\n"
        f"Median: {median_val:.2f} {unit}\n"
        f"Std: {std_val:.2f} {unit}\n"
        f"Min: {min_val} {unit}\n"
        f"Max: {max_val} {unit}\n"
        f"Skew: {skew_val:.2f}"
    )

    # Place statistics inside the plot (top-right corner)
    plt.text(
        0.95, 0.95, stats_text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        multialignment='right',
        fontfamily='monospace',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    )

    plt.tight_layout()
    plt.show()


def make_histogram(dataframe, feature, bins = 30, custom_ticks=None, unit=''):

    # Size of the figure
    plt.figure(figsize=(16, 9))

    # Plot the histogram
    plt.hist(dataframe[feature].dropna(), bins=bins, density=True, rwidth=0.95)

    # Set axis labels
    plt.xlabel(feature)
    plt.ylabel('Relative Frequency')

    # Custom x-axis ticks
    if custom_ticks is not None:
        plt.xticks(custom_ticks)

    # Control the lines of the grid
    plt.grid(axis="x", visible=False)
    plt.grid(axis="y", visible=True)

    # Set the ticks
    plt.tick_params(axis='x', pad=10)
    plt.tick_params(axis='y', pad=5)

    # Statistics
    mean_val = dataframe[feature].mean()
    std_val = dataframe[feature].std()
    median_val = dataframe[feature].median()
    min_val = dataframe[feature].min()
    max_val = dataframe[feature].max()
    skew_val = dataframe[feature].skew()

    stats_text = (
        f"Mean: {mean_val:.2f} {unit}\n"
        f"Median: {median_val:.2f} {unit}\n"
        f"Std: {std_val:.2f} {unit}\n"
        f"Min: {min_val:.2f} {unit}\n"
        f"Max: {max_val:.2f} {unit}\n"
        f"Skew: {skew_val:.2f}"
    )

    # Place statistics inside the plot (top-right corner)
    plt.text(
        0.95, 0.95, stats_text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        multialignment='right',
        fontfamily='monospace',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    )

    # Show the graphic
    plt.tight_layout()
    plt.show()


def make_boxplot(dataframe, feature, unit=''):

    # Size of the figure
    plt.figure(figsize=(16,9))

    # Plot the boxplot
    sns.boxplot(dataframe, x=feature)

    # Set axis labels
    plt.xlabel(feature)
    plt.ylabel("Values")

    # Statistics
    mean_val = dataframe[feature].mean()
    std_val = dataframe[feature].std()
    median_val = dataframe[feature].median()
    min_val = dataframe[feature].min()
    max_val = dataframe[feature].max()
    skew_val = dataframe[feature].skew()

    stats_text = (
        f"Mean: {mean_val:.2f} {unit}\n"
        f"Median: {median_val:.2f} {unit}\n"
        f"Std: {std_val:.2f} {unit}\n"
        f"Min: {min_val:.2f} {unit}\n"
        f"Max: {max_val:.2f} {unit}\n"
        f"Skew: {skew_val:.2f}"
    )

    # Place statistics inside the plot (top-right corner)
    plt.text(
        0.95, 0.95, stats_text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        multialignment='right',
        fontfamily='monospace',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    )

    # Show the graphic
    plt.tight_layout()
    plt.show()


# Functions for multivariate analysis
#----------------------------------------------------------------------------------

def make_heat_map(dataframe, features):

    # Set the figure size
    plt.figure(figsize=(10, 6))

    # Compute correlation matrix
    corr = dataframe[features].corr()

    # Creating a mask to hide the upper triangle of the heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Plot the heatmap
    sns.set_theme(font_scale=1.2)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", mask=mask)

    # Show the graphic
    plt.tight_layout()
    plt.show()


def make_stacked_barplots(dataframe, cat_vars):

    # Validate at least two categorical variables are provided
    if len(cat_vars) < 2:
        raise ValueError("You must provide at least two categorical variables.")

    # Generate all pairwise permutations of categorical variables
    for var1, var2 in itertools.permutations(cat_vars, 2):

        # Contingency table normalized by rows (percentages)
        crosstab = pd.crosstab(dataframe[var1], dataframe[var2], normalize="index") * 100

        # Set the figure size
        plt.figure(figsize=(16, 9))

        # Stacked barplot
        ax = crosstab.plot(kind="bar", stacked=True, figsize=(16, 9))

        plt.ylabel("Percentage (%)")
        plt.xlabel(var1)

        # Add text labels inside bars
        for i, row in enumerate(crosstab.values):
            cumulative = 0
            for j, value in enumerate(row):
                if value > 0:  # avoid writing on empty segments
                    ax.text(
                        i,
                        cumulative + value / 2,
                        f"{value:.2f}%",
                        ha="center", va="center",
                        color="white", fontweight="bold", fontsize=9
                    )
                cumulative += value

        # Plot legend
        plt.legend(title=var2, bbox_to_anchor=(1.05, 1), loc="upper left")

        # Control the lines of the grid
        plt.grid(axis="x", visible=False)
        plt.grid(axis="y", visible=True)

        # Rotate the x-axis labels
        plt.xticks(rotation=0)

        # Show the graphic
        plt.tight_layout()
        plt.show()


def make_grouped_boxplots(dataframe, num_vars, cat_vars, type_plot='boxplot', unit=''):

    for num_var, cat_var in itertools.product(num_vars, cat_vars):

        # Set the figure size
        plt.figure(figsize=(16, 9))

        # Boxplot or Violinplot
        if type_plot == 'boxplot':
            ax = sns.boxplot(data=dataframe, x=cat_var, y=num_var)
        else:
            ax = sns.violinplot(data=dataframe, x=cat_var, y=num_var, box=None)

        # Labels
        plt.xlabel(cat_var)
        plt.ylabel(f"{num_var} ({unit})" if unit else num_var)

        # Compute stats per category
        grouped_stats = dataframe.groupby(cat_var)[num_var].agg(
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max",
            skew="skew"
        )

        # Format stats vertically
        stats_lines = [""]
        for cat, row in grouped_stats.iterrows():
            stats_lines.append(f"{cat}:")
            stats_lines.append(f"  Mean: {row['mean']:.2f}{unit}")
            stats_lines.append(f"  Median: {row['median']:.2f}{unit}")
            stats_lines.append(f"  Std: {row['std']:.2f}")
            stats_lines.append(f"  Min: {row['min']:.2f}")
            stats_lines.append(f"  Max: {row['max']:.2f}")
            stats_lines.append(f"  Skew: {row['skew']:.2f}")
            stats_lines.append("")
        stats_text = "\n".join(stats_lines)

        # Place stats box outside (to the right of the plot)
        plt.gcf().subplots_adjust(right=0.7)
        plt.text(
            1.01, 1, stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            horizontalalignment='left',
            multialignment='left',
            fontfamily='monospace',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )

        # Show the graphic
        plt.tight_layout()
        plt.show()


def make_scatter_matrix(df, num_vars):

    # Set figure size
    plt.figure(figsize=(16, 9))

    # Make the scatterplot matrix
    g = sns.pairplot(df[num_vars], diag_kind='kde', corner=False)

    # Adjust y-axis labels
    for ax in g.axes.flatten():
        if ax is not None:
            ax.yaxis.label.set_rotation(0)
            ax.yaxis.label.set_ha('right')
            ax.yaxis.labelpad = 13

    # Adjust layout to avoid overlapping
    g.figure.tight_layout()

    plt.show()
