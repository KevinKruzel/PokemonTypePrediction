import streamlit as st
import pandas as pd

def apply_pokemon_filters(df: pd.DataFrame):
    st.sidebar.header("Filters")

    include_legendary = st.sidebar.checkbox(
        "Include Legendary Pokémon",
        value=True,
        key="include_legendary"
    )

    include_non_fully_evolved = st.sidebar.checkbox(
        "Include Non-Fully Evolved Pokémon",
        value=True,
        key="include_non_fully_evolved"
    )

    include_regional_variants = st.sidebar.checkbox(
        "Include Regional Variants",
        value=True,
        key="include_regional_variants"
    )

    include_dual_typed = st.sidebar.checkbox(
        "Include Dual-Typed Pokémon",
        value=True,
        key="include_dual_typed"
    )

    min_gen = int(df["generation"].min())
    max_gen = int(df["generation"].max())

    gen_min, gen_max = st.sidebar.slider(
        "Generation Range",
        min_value=min_gen,
        max_value=max_gen,
        value=(min_gen, max_gen),
        help="Filter Pokémon by generation number.",
        key="generation_range",
    )

    df_filtered = df.copy()
    df_filtered = df_filtered[
        (df_filtered["generation"] >= gen_min) &
        (df_filtered["generation"] <= gen_max)
    ]

    if not include_legendary:
        df_filtered = df_filtered[df_filtered["is_legendary"] == False]

    if not include_non_fully_evolved:
        df_filtered = df_filtered[df_filtered["is_fully_evolved"] == True]

    if not include_regional_variants:
        df_filtered = df_filtered[df_filtered["is_regional_variant"] == False]

    if not include_dual_typed:
        df_filtered = df_filtered[df_filtered["is_mono-type"] == True]

    st.caption(f"Current filters: {len(df_filtered)} Pokémon selected.")
    return df_filtered
