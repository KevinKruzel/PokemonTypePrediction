import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from filters import apply_pokemon_filters
from filters import TYPE_COLORS

# ───────────────────────────
# PAGE CONFIG
# ───────────────────────────
st.set_page_config(
    page_title="EDA Gallery",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────
# LOAD DATA
# ───────────────────────────
DATA_PATH = Path(__file__).parent.parent / "data" / "pokemon_dataset.csv"
df = pd.read_csv(DATA_PATH)

# Apply filters from sidebar
df_filtered = apply_pokemon_filters(df)

# ───────────────────────────
# PAGE HEADER
# ───────────────────────────
st.title("📊 Exploratory Data Analysis Gallery")
st.caption(f"Current filters: {len(df_filtered)} Pokémon selected.")
st.markdown("""
This page serves the purpose of allowing us to explore the data first and make an informed decision about the type of prediction model
we want to use to determine a Pokémon's type given its six base stats: HP, Attack, Defense, Special Attack, Special Defense, and Speed.
""")
st.divider()

# ───────────────────────────
# BOXPLOT FUNCTION (To be used six times on this page)
# ───────────────────────────
def stat_boxplot(container, df_filtered, stat_col, stat_label):
    with container:
        if df_filtered.empty:
            st.warning("No Pokémon available for the selected filters.")
        else:
            # Define the order the boxplots will be arranged, which is in ascending order by their mean
            type_order = (
                df_filtered.groupby("primary_type")[stat_col]
                .mean()
                .reset_index()
                .sort_values(stat_col)["primary_type"]
                .tolist()
            )

            # Calculate the overall mean of all Pokemon
            overall_mean = df_filtered[stat_col].mean()

            # Create the boxplot chart
            fig_box = px.box(
                df_filtered,
                x="primary_type",
                y=stat_col,
                category_orders={"primary_type": type_order},
                title=f"{stat_label} Stat Distribution by Primary Type",
                color="primary_type",
                color_discrete_map=TYPE_COLORS,
            )

            # Make the color of each boxplot correspond the group's type
            for trace in fig_box.data:
                t = trace.name
                c = TYPE_COLORS.get(t, "#808080")
                trace.update(
                    marker_color=c,
                    marker_line_color=c,
                    line_color=c,
                )

            # Add the overall mean line to the chart
            fig_box.add_shape(
                type="line",
                x0=-0.5,
                x1=len(type_order),
                y0=overall_mean,
                y1=overall_mean,
                line=dict(color="red", width=2, dash="dash"),
            )

            # Add the label of the overall mean line to the chart
            fig_box.add_annotation(
                x=len(type_order)-0.5,
                y=overall_mean,
                text=f"Mean: {overall_mean:.1f}",
                showarrow=False,
                yshift=10,
                font=dict(color="red"),
            )

            # Adjust features of the chart
            fig_box.update_layout(
                xaxis_title="Primary Type",
                yaxis_title=stat_label,
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False,
            )

            # Display the chart
            st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})

# ───────────────────────────
# ROW 1
# ───────────────────────────
st.subheader("Exploring the Amount of Pokémon by Type")
st.markdown("""
For those who are unaware, Pokémon can have one or two types. The initial goal of the project was to be able to predict a Pokémon's typing (so if a Pokémon was dual-typed,
the model would be able to predict both). However, for simplicity's sake and for easier visualization, we will only be evaluating Pokémon by their **primary** type, which is the first one listed.
For those who are curious about the distribution of Pokémon by their primary and secondary type, it can be visualized in the heatmap below.
<br>
The distribution of Pokémon by only their primary type can be see in the bar chart below. The key observation about this that can be made is the large
disparity between the amount of types. This inbalance makes certain prediction models like K-Nearest Neighbors unsuitable.
""", unsafe_allow_html=True)
st.divider()

col_heatmap, col_bar = st.columns([3, 2])  # 3/5 width and 2/5 width

# Heatmap that shows the amount of Pokemon in each primary and secondary type combination
with col_heatmap:
    # Check first if there are no Pokemon to chart
    if df_filtered.empty:
        st.warning("No Pokémon available for the selected filters.")
    else:
        temp = df_filtered.copy()

        temp["secondary_type_display"] = temp["secondary_type"]
        mono_mask = temp["secondary_type_display"].isna()
        temp.loc[mono_mask, "secondary_type_display"] = temp.loc[mono_mask, "primary_type"]

        # Group all Pokemon by their primary type, and then by their secondary type
        type_counts = (
            temp.groupby(["primary_type", "secondary_type_display"])["pokemon_id"]
            .count()
            .reset_index(name="count")
        )

        # Create the table that will be used for the heatmap
        pivot_table = type_counts.pivot_table(
            index="secondary_type_display",
            columns="primary_type",
            values="count",
            fill_value=0,
        )

        # Sort the table so that the types are in alphabetical order
        pivot_table = pivot_table.reindex(
            index=sorted(pivot_table.index),
            columns=sorted(pivot_table.columns),
        )

        # Create the heatmap
        fig = px.imshow(
            pivot_table,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Reds",
            labels=dict(color="Number of Pokémon"),
        )

        # Update features about the heatmap
        fig.update_layout(
            title="Heatmap Showing the Amount of Pokémon With Each Type Combination",
            xaxis_title="Primary Type",
            yaxis_title="Secondary Type",
            margin=dict(l=10, r=10, t=40, b=10),
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# Bar chart that displays the amount of Pokemon in each primary type group
with col_bar:
    # Check first if there are no Pokemon to chart
    if df_filtered.empty:
        st.warning("No Pokémon available for the selected filters.")
    else:
        # Group Pokemon by their primary type and then sort them in alphabetical order
        type_counts_bar = (
            df_filtered.groupby("primary_type")["pokemon_id"]
            .count()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

        # Create the bar chart
        fig_bar = px.bar(
            type_counts_bar,
            x="primary_type",
            y="count",
            title="Pokémon Count by Primary Type",
            color="primary_type",
            color_discrete_map=TYPE_COLORS,
            text_auto=True,
        )

        # Update other features of the bar chart
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            xaxis_title="Primary Type",
            yaxis_title="Number of Pokémon",
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
        )

        # Display the bar chart
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
        
st.divider()
# ───────────────────────────
# ROW 2-4 Boxplots
# ───────────────────────────
st.subheader("Visualizing the Difference in the Base Stats of Pokémon by Primary Type")
st.markdown("""
The underlying assumption about employing this model for the purposes of predicting a Pokémon's type using it's base stats is that each type of Pokémon has distinct differences.
The boxplots created below aim to visual these differences between types. Each plot represents one of the base stats, and includes a dashed red line that serves the purpose of
representing the overall mean value of that stat across all Pokémon in the filtered dataset. The types are sorted in ascending order by their mean value.
After looking at the boxplots, it is clear to see that while there is significant overlap between types, there is sufficient evidence to say that each type is distinct enough to warrant using our machine learning model.
""")
st.divider()

col1_r2, col2_r2 = st.columns(2)
stat_boxplot(col1_r2, df_filtered, "hp", "HP")
stat_boxplot(col2_r2, df_filtered, "attack", "Attack")

col1_r3, col2_r3 = st.columns(2)
stat_boxplot(col1_r3, df_filtered, "defense", "Defense")
stat_boxplot(col2_r3, df_filtered, "special-attack", "Special Attack")

col1_r4, col2_r4 = st.columns(2)
stat_boxplot(col1_r4, df_filtered, "special-defense", "Special Defense")
stat_boxplot(col2_r4, df_filtered, "speed", "Speed")

st.divider()

st.subheader("Visualizing Physical and Capture Characteristics by Primary Type")
st.markdown("""
In addition to the six base stats, Pokémon have other numeric characteristics that may also relate to their typing,
such as height, weight, capture rate, and overall total stats. The boxplots below show how these quantities are distributed
across primary types, again sorted by mean value and including an overall mean reference line.
""")

st.divider()

col1_r5, col2_r5 = st.columns(2)
stat_boxplot(col1_r5, df_filtered, "height", "Height")
stat_boxplot(col2_r5, df_filtered, "weight", "Weight")

col1_r6, col2_r6 = st.columns(2)
stat_boxplot(col1_r6, df_filtered, "capture_rate", "Capture Rate")
stat_boxplot(col2_r6, df_filtered, "total_stats", "Total Stats")

st.divider()

# ───────────────────────────
# ROW 5 - COLOR
# ───────────────────────────
st.subheader("Exploring Pokémon Color by Primary Type")
st.markdown("""
Pokémon are categorized by a **color** attribute in the Pokédex. While it is not directly used for prediction,
it reveals interesting relationships between how Game Freak conceptually links color palettes and type design.
The heatmap shows the frequency of each color within each primary type, and the bar chart summarizes overall color distribution.
""")
st.divider()

col_color_heatmap, col_color_bar = st.columns([3, 2])

# Color → Hex mapping (Pokédex standard colors)
POKEDEX_COLOR_MAP = {
    "black": "#000000",
    "blue": "#3B4CCA",
    "brown": "#8B4513",
    "gray": "#A8A8A8",
    "green": "#4CAF50",
    "pink": "#FF69B4",
    "purple": "#A040A0",
    "red": "#FF0000",
    "white": "#FFFFFF",
    "yellow": "#FFD700",
}

with col_color_heatmap:
    if df_filtered.empty:
        st.warning("No Pokémon available for the selected filters.")
    else:
        color_counts = (
            df_filtered
            .groupby(["color", "primary_type"])["pokemon_id"]
            .count()
            .reset_index(name="count")
        )

        pivot_color = color_counts.pivot_table(
            index="color",
            columns="primary_type",
            values="count",
            fill_value=0,
        )

        # Sort rows and columns alphabetically like the earlier heatmap
        pivot_color = pivot_color.reindex(
            index=sorted(pivot_color.index),
            columns=sorted(pivot_color.columns),
        )

        fig_color_heat = px.imshow(
            pivot_color,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Reds",   # same as earlier heatmap
            labels=dict(color="Number of Pokémon"),
        )

        fig_color_heat.update_layout(
            title="Heatmap of Pokémon Colors by Primary Type",
            xaxis_title="Primary Type",
            yaxis_title="Color",
            margin=dict(l=10, r=10, t=40, b=10),
        )

        st.plotly_chart(fig_color_heat, use_container_width=True, config={"displayModeBar": False})

with col_color_bar:
    if df_filtered.empty:
        st.warning("No Pokémon available for the selected filters.")
    else:
        color_counts_bar = (
            df_filtered
            .groupby("color")["pokemon_id"]
            .count()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

        # Map each bar to its color using the hex mapping
        color_counts_bar["color_hex"] = color_counts_bar["color"].map(
            lambda c: POKEDEX_COLOR_MAP.get(c.lower(), "#808080")
        )

        fig_color_bar = px.bar(
            color_counts_bar,
            x="color",
            y="count",
            title="Pokémon Count by Color",
            text_auto=True,
            color="color",  # use color names as categories
            color_discrete_map=POKEDEX_COLOR_MAP,
        )

        fig_color_bar.update_traces(textposition="outside")
        fig_color_bar.update_layout(
            xaxis_title="Color",
            yaxis_title="Number of Pokémon",
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
        )

        st.plotly_chart(fig_color_bar, use_container_width=True, config={"displayModeBar": False})

st.divider()

# ───────────────────────────
# ROW 6- EGG GROUPS
# ───────────────────────────
st.subheader("Exploring Pokémon Egg Groups")
st.markdown("""
Egg groups describe which Pokémon can breed with each other. Many Pokémon belong to one egg group, while others
belong to two, allowing them to act as “bridges” between groups. The visualizations below explore how egg groups
relate to each other and to primary types.

- The **top-left heatmap** shows how often pairs of egg groups occur together (Egg Group 1 vs Egg Group 2).
- The **bar chart on the right** shows how many Pokémon belong to each egg group (Pokémon in two groups count once in each).
- The **bottom-left heatmap** shows how Pokémon are distributed across **primary type** and **egg group**.
""")
st.divider()

EGG_GROUP_COLORS = {
    "monster": "#8B0000",
    "water1": "#1E90FF",
    "water2": "#4169E1",
    "water3": "#0000CD",
    "bug": "#7FFF00",
    "flying": "#87CEEB",
    "field": "#CD853F",
    "fairy": "#FF69B4",
    "grass": "#228B22",
    "humanlike": "#800080",
    "mineral": "#708090",
    "amorphous": "#A0522D",
    "ditto": "#BA55D3",
    "dragon": "#9932CC",
    "undiscovered": "#696969",
}

if df_filtered.empty:
    st.warning("No Pokémon available for the selected filters.")
else:
    # Prepare a melted egg-group membership table for reuse
    egg_temp = df_filtered[["pokemon_id", "primary_type", "egg_group_1", "egg_group_2"]].copy()

    egg_melted = egg_temp.melt(
        id_vars=["pokemon_id", "primary_type"],
        value_vars=["egg_group_1", "egg_group_2"],
        value_name="egg_group"
    )
    egg_melted = egg_melted.dropna(subset=["egg_group"])

    # Avoid double-counting if egg_group_1 == egg_group_2
    egg_melted = egg_melted.drop_duplicates(subset=["pokemon_id", "egg_group"])

    col_egg_heatmap, col_egg_bar = st.columns([3, 2])

    with col_egg_heatmap:
        # Top heatmap: Egg Group 1 (x) vs Egg Group 2 (y)
        egg_pair_counts = (
            df_filtered
            .groupby(["egg_group_1", "egg_group_2"])["pokemon_id"]
            .nunique()
            .reset_index(name="count")
        )

        pivot_egg_pair = egg_pair_counts.pivot_table(
            index="egg_group_2",
            columns="egg_group_1",
            values="count",
            fill_value=0,
        )

        # Sort rows and columns alphabetically
        pivot_egg_pair = pivot_egg_pair.reindex(
            index=sorted(pivot_egg_pair.index),
            columns=sorted(pivot_egg_pair.columns),
        )

        fig_egg_pair = px.imshow(
            pivot_egg_pair,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Reds",
            labels=dict(color="Number of Pokémon"),
        )

        fig_egg_pair.update_layout(
            title="Heatmap of Egg Group 2 vs Egg Group 1",
            xaxis_title="Egg Group 1",
            yaxis_title="Egg Group 2",
            margin=dict(l=10, r=10, t=40, b=10),
        )

        st.plotly_chart(fig_egg_pair, use_container_width=True, config={"displayModeBar": False})

        # Bottom heatmap: Egg Group (y) vs Primary Type (x), counting multi-group Pokémon in each group
        egg_type_counts = (
            egg_melted
            .groupby(["egg_group", "primary_type"])["pokemon_id"]
            .nunique()
            .reset_index(name="count")
        )

        pivot_egg_type = egg_type_counts.pivot_table(
            index="egg_group",
            columns="primary_type",
            values="count",
            fill_value=0,
        )

        pivot_egg_type = pivot_egg_type.reindex(
            index=sorted(pivot_egg_type.index),
            columns=sorted(pivot_egg_type.columns),
        )

        fig_egg_type = px.imshow(
            pivot_egg_type,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Reds",
            labels=dict(color="Number of Pokémon"),
        )

        fig_egg_type.update_layout(
            title="Heatmap of Egg Group by Primary Type",
            xaxis_title="Primary Type",
            yaxis_title="Egg Group",
            margin=dict(l=10, r=10, t=40, b=10),
        )

        st.plotly_chart(fig_egg_type, use_container_width=True, config={"displayModeBar": False})

    with col_egg_bar:
        egg_group_counts = (
            egg_melted
            .groupby("egg_group")["pokemon_id"]
            .nunique()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

        # Color each bar according to its egg group
        fig_egg_bar = px.bar(
            egg_group_counts,
            x="egg_group",
            y="count",
            title="Pokémon Count by Egg Group",
            text_auto=True,
            color="egg_group",
            color_discrete_map=EGG_GROUP_COLORS,
        )

        fig_egg_bar.update_traces(textposition="outside")
        fig_egg_bar.update_layout(
            xaxis_title="Egg Group",
            yaxis_title="Number of Pokémon",
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
        )

        st.plotly_chart(fig_egg_bar, use_container_width=True, config={"displayModeBar": False})

st.divider()

# ───────────────────────────
# ROW 7 – Controls for Scatterplot and Scatterplot
# ───────────────────────────
st.subheader("Customizable Scatterplot")
st.markdown("""
The controls for creating a custom scatterplot are given below to further help visualize the differences in Pokémon types.
Users can select any amount of type groups and two base stats to compare the distribution of these values. You can also compare type group averages as well.

<br>

One of the clearest and biggest examples of two types being distinctly different is presented in the default values selected here: The distribution of fighting and psychic Pokémon's
attack and special attack stats. Fighting type Pokémon have distinctly high attack and low special attack, whereas psychic type Pokémon are the opposite.
""", unsafe_allow_html=True)
st.divider()

col1_r5, col2_r5 = st.columns([1, 2])  # 1/3 width and 2/3 width

# Scatterplot Controls
with col1_r5:
    st.subheader("Scatterplot Controls")

    # Map the "pretty" to the "ugly" type names
    stat_labels = {
        "HP": "hp",
        "Attack": "attack",
        "Defense": "defense",
        "Special Attack": "special-attack",
        "Special Defense": "special-defense",
        "Speed": "speed",
        "Height": "height",
        "Weight": "weight",
        "Capture Rate": "capture_rate",
        "Total Stats": "total_stats",
    }

    # Selectbox to decide which type of points to graph, defaults to "Individual Pokémon Only"
    display_mode = st.selectbox(
        "Points to Display",
        [
            "Individual Pokémon Only",
            "Type Averages Only",
            "Both Individuals and Averages",
        ],
        index=0,
        help="Individual Pokémon will appear as small data points. Type averages will appear as large data points."
    )

    # Selectboxes to decide which of the six stats to represent the two axes of the scatterplot, defaults to "Attack" and "Special Attack"
    x_label = st.selectbox(
        "X-Axis Stat",
        list(stat_labels.keys()),
        index=list(stat_labels.keys()).index("Attack")
    )
    y_label = st.selectbox(
        "Y-Axis Stat",
        list(stat_labels.keys()),
        index=list(stat_labels.keys()).index("Special Attack")
    )

    # Use the naming map to make the "ugly" stat names "pretty"
    x_stat = stat_labels[x_label]
    y_stat = stat_labels[y_label]

    # Checkbox to plot a y = x line on the chart, defaults to False
    show_diag_line = st.checkbox(
        "Show Y = X Reference Line",
        value=False,
        help="When checked, draws a diagonal line where the two stats are equal."
    )

    # Subsection of the controls that includes checkboxes to decide which type groups are included
    st.markdown("**Types to Include**")
    type_col1, type_col2, type_col3 = st.columns(3)

    # Create a checkbox for each type, arranged into 3 columns alphabetically, defaults to having Psychic and Fighting being set to True
    sorted_types = sorted(TYPE_COLORS.keys())
    third = len(sorted_types) // 3
    selected_types = []

    with type_col1:
        for t in sorted_types[:third]:
            default_checked = (t in ["fighting", "psychic"])
            if st.checkbox(t.capitalize(), value=default_checked, key=f"type_{t}"):
                selected_types.append(t)
    with type_col2:
        for t in sorted_types[third: third*2]:
            default_checked = (t in ["fighting", "psychic"])
            if st.checkbox(t.capitalize(), value=default_checked, key=f"type_{t}_2"):
                selected_types.append(t)
    with type_col3:
        for t in sorted_types[third*2:]:
            default_checked = (t in ["fighting", "psychic"])
            if st.checkbox(t.capitalize(), value=default_checked, key=f"type_{t}_3"):
                selected_types.append(t)

# Scatterplot
with col2_r5:
    st.subheader("Stat vs Stat Scatterplot")

    # Display an error message if there are no Pokemon to chart given the selected filters
    if df_filtered.empty:
        st.warning("No Pokémon available for the selected filters.")
    else:
        df_scatter = df_filtered.copy()

        # Set the selected types given the status of the checkboxes
        if selected_types:
            df_scatter = df_scatter[df_scatter["primary_type"].isin(selected_types)]
        else:
            df_scatter = df_scatter.iloc[0:0]

        # Display an error message if there are no Pokemon to chart given the status of the checkboxes
        if df_scatter.empty:
            st.warning("No Pokémon match the selected types.")
        else:

            # Section of code to chart indiviudal Pokemon, will activate if "both" is selected as well
            if display_mode in ("Individual Pokémon Only", "Both Individuals and Averages"):
                fig_scatter = px.scatter(
                    df_scatter,
                    x=x_stat,
                    y=y_stat,
                    color="primary_type",
                    color_discrete_map=TYPE_COLORS,
                    title=f"{x_label} vs {y_label} by Primary Type",
                    hover_name="pokemon_name",
                    hover_data={
                        "primary_type": True,
                        "generation": True,
                        x_stat: True,
                        y_stat: True,
                        "pokemon_id": True,
                    },
                )
            else:
                type_means = (
                    df_scatter
                    .groupby("primary_type", as_index=False)[[x_stat, y_stat]]
                    .mean()
                )

                fig_scatter = px.scatter(
                    type_means,
                    x=x_stat,
                    y=y_stat,
                    color="primary_type",
                    color_discrete_map=TYPE_COLORS,
                    title=f"{x_label} vs {y_label} – Type Averages",
                    hover_name="primary_type",
                    hover_data={
                        "primary_type": True,
                        x_stat: True,
                        y_stat: True,
                    },
                )

            # Section of code to chart type averages, will activate if "both" is selected as well
            if display_mode in ("Type Averages Only", "Both Individuals and Averages"):
                type_means = (
                    df_scatter
                    .groupby("primary_type", as_index=False)[[x_stat, y_stat]]
                    .mean()
                )

                for _, row in type_means.iterrows():
                    t = row["primary_type"]
                    c = TYPE_COLORS.get(t, "#808080")
                    fig_scatter.add_scatter(
                        x=[row[x_stat]],
                        y=[row[y_stat]],
                        mode="markers",
                        marker=dict(
                            size=18,
                            color=c,
                            line=dict(color="black", width=1.5),
                        ),
                        name=f"{t} avg",
                        showlegend=False,
                    )

            # Display the y = x line if it was selected in the checkbox
            if show_diag_line:
                df_scatter[x_stat] = pd.to_numeric(df_scatter[x_stat], errors="coerce")
                df_scatter[y_stat] = pd.to_numeric(df_scatter[y_stat], errors="coerce")
                df_scatter_clean = df_scatter.dropna(subset=[x_stat, y_stat])

                if not df_scatter_clean.empty:
                    min_val = min(
                        df_scatter_clean[x_stat].min(),
                        df_scatter_clean[y_stat].min()
                    )
                    max_val = max(
                        df_scatter_clean[x_stat].max(),
                        df_scatter_clean[y_stat].max()
                    )

                    fig_scatter.add_shape(
                        type="line",
                        x0=min_val,
                        y0=min_val,
                        x1=max_val,
                        y1=max_val,
                        line=dict(color="gray", dash="dash"),
                        layer="above",
                        name="y = x",
                    )

            # Adjust features of the chart
            fig_scatter.update_layout(
                xaxis_title=x_label,
                yaxis_title=y_label,
                margin=dict(l=10, r=10, t=40, b=10),
                legend_title="Primary Type",
            )

            # Display the scatterplot chart
            st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})

st.divider()
st.caption("Data was collected using pokeapi found here: https://pokeapi.co/")

with st.expander("Data Preview"):
    st.dataframe(df)
    
csv_data = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Raw Data (CSV)",
    data=csv_data,
    file_name="pokemon_dataset.csv",
    mime="text/csv",
)
