import pandas as pd
import numpy as np
import plotly.express as px


def kpi_metrics(df):

    numeric = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_columns": len(numeric),
        "missing_cells": int(
            df.isna().sum().sum()
        ),
        "duplicate_rows": int(
            df.duplicated().sum()
        )
    }


def histogram(df, column, bins=30):

    data = df[[column]].dropna()

    fig = px.histogram(
        data,
        x=column,
        nbins=bins,
        title=f"Distribution — {column}"
    )

    fig.update_layout(
        height=500,
        margin=dict(
            l=40,
            r=40,
            t=70,
            b=40
        )
    )

    return fig


def box_plot(df, column):

    data = df[[column]].dropna()

    fig = px.box(
        data,
        y=column,
        points="outliers",
        title=f"Box Plot — {column}"
    )

    fig.update_layout(
        height=500
    )

    return fig


def scatter_plot(
    df,
    x,
    y,
    color=None
):

    columns = [x, y]

    if color and color not in columns:

        columns.append(color)

    data = df[columns].dropna()

    fig = px.scatter(
        data,
        x=x,
        y=y,
        color=color,
        title=f"{y} vs {x}",
        trendline="ols"
    )

    fig.update_layout(
        height=550
    )

    return fig


def line_chart(
    df,
    x,
    y
):

    data = df[[x, y]].dropna()

    data = data.sort_values(x)

    fig = px.line(
        data,
        x=x,
        y=y,
        markers=True,
        title=f"{y} Trend"
    )

    fig.update_layout(
        height=500
    )

    return fig


def bar_chart(
    df,
    category,
    value,
    aggregation="mean"
):

    data = df[
        [category, value]
    ].dropna()

    if aggregation == "sum":

        grouped = (
            data
            .groupby(category)[value]
            .sum()
            .reset_index()
        )

    elif aggregation == "count":

        grouped = (
            data
            .groupby(category)[value]
            .count()
            .reset_index()
        )

    else:

        grouped = (
            data
            .groupby(category)[value]
            .mean()
            .reset_index()
        )

    grouped = grouped.sort_values(
        value,
        ascending=False
    )

    fig = px.bar(
        grouped,
        x=category,
        y=value,
        title=f"{value} by {category}"
    )

    fig.update_layout(
        height=500
    )

    return fig


def correlation_heatmap(df):

    numeric = df.select_dtypes(
        include=np.number
    )

    if numeric.shape[1] < 2:

        return None

    corr = numeric.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Agricultural Feature Correlation",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )

    fig.update_layout(
        height=max(
            500,
            len(corr.columns) * 35
        )
    )

    return fig