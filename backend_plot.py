# Standard library imports
import datetime
import os

# Third party imports
import dask
import numpy as np
import pandas as pd
import yaml

# Configure dask
dask.config.set({"dataframe.query-planning": True})

# Relative imports from third party libraries
from bokeh.io import output_file, save
from bokeh.models import (
    CustomJS,
    DatetimeTickFormatter,
    FixedTicker,
    HoverTool,
    LinearAxis,
    Range1d,
    TapTool,
)
from bokeh.plotting import figure
from dask import dataframe as dd


def load_fills_filtered(
    path_parquet: str, start_time: datetime.datetime, end_time: datetime.datetime
) -> np.ndarray:
    # read the parquet file
    df_fills = pd.read_parquet(path_parquet)

    # Ensure no fill is missing
    df_fills.sort_index(inplace=True)
    uniques_fills = np.array(df_fills.index.unique())
    assert len(set((uniques_fills[1:] - uniques_fills[:-1]))) == 1

    # convert t_start to unixtime in nanoseconds
    t_start = pd.Timestamp(start_time).timestamp() * 1e9
    t_stop = pd.Timestamp(end_time).timestamp() * 1e9

    fills_filtered = df_fills[
        (df_fills["tsStart"] > t_start) & (df_fills["tsEnd"] < t_stop)
    ].index.unique()

    return fills_filtered


def add_fill_metadata(
    relative_path_metadata: str, tag_files: list, dict_fills: dict, fill_idx: int
) -> dict:
    # open file in weekly_follow_up
    l_tag_filename = [kk for kk in tag_files if str(fill_idx) in kk]

    # if file exists, read it and extract tags and comment
    filename = f"{relative_path_metadata}{l_tag_filename[0]}"
    if not os.path.exists(filename):
        print(f"File {filename} does not exist")
    else:
        with open(f"{relative_path_metadata}{l_tag_filename[0]}") as f:
            yaml_file = yaml.load(f, Loader=yaml.FullLoader)

            dict_fills[f"{fill_idx}"]["df"]["start"] = yaml_file["start"]
            dict_fills[f"{fill_idx}"]["df"]["end"] = yaml_file["end"]
            dict_fills[f"{fill_idx}"]["df"]["duration"] = yaml_file["duration"]

            if yaml_file["tags"] is None:
                dict_fills[f"{fill_idx}"]["df"]["tags"] = ""
            else:
                dict_fills[f"{fill_idx}"]["df"]["tags"] = ", ".join(
                    [str(tags) for tags in yaml_file["tags"]]
                )
            if yaml_file["comment"] is None:
                dict_fills[f"{fill_idx}"]["df"]["comment"] = ""
            else:
                dict_fills[f"{fill_idx}"]["df"]["comment"] = ", ".join(
                    [str(comment) for comment in yaml_file["comment"]]
                )

    return dict_fills


def get_dict_fills_data(
    fills_filtered: np.ndarray,
    path_raw_data: str,
    path_tag_files: str,
    variables: list,
    verbose: bool = True,
):
    dict_fills = {}

    # list all files in the weekly_follow_up directory
    tag_files = os.listdir(path_tag_files)
    for fill_idx in fills_filtered:
        try:
            ddf = dd.read_parquet(
                f"{path_raw_data}HX:FILLN={fill_idx}",
                engine="pyarrow",
                columns=variables,
            )

            # Sample the dataframe
            sample_df = ddf.compute().sort_index().ffill().dropna().sample(frac=0.001).sort_index()

            # convert sample_df index in human readable format
            sample_df.index = pd.to_datetime(sample_df.index, unit="ns")
            dict_fills[f"{fill_idx}"] = {}
            dict_fills[f"{fill_idx}"]["fill"] = fill_idx
            dict_fills[f"{fill_idx}"]["df"] = sample_df
            dict_fills[f"{fill_idx}"]["df"]["fill"] = fill_idx
            # ! Update link accordingly
            dict_fills[f"{fill_idx}"]["df"]["link"] = "https://lhc-fills.web.cern.ch/fills/" + str(
                fill_idx
            )

            # Add metadata
            dict_fills = add_fill_metadata(path_tag_files, tag_files, dict_fills, fill_idx)
            if verbose:
                print("Fill ", fill_idx, " done")

        except Exception as e:
            if verbose:
                print("File not found for fill or corrupted data ", fill_idx)
                print(e)

    return dict_fills


def plot_fill_data(
    dict_fills: dict,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    dict_var: dict,
    color_energy_even="yellowgreen",
    color_energy_odd="peru",
    clickable_link=True,
    save_html: bool = True,
    html_filepath: str = "plot.html",
):
    p = figure(title=f"DL2 Report from {start_time} to {end_time}")

    # set bokeh figure size
    p.width = 1000
    p.height = 300

    # Configure axes
    date_format = "%a, %d %b"
    p.xaxis[0].formatter = DatetimeTickFormatter(
        days=date_format,
        months=date_format,
        years=date_format,
    )
    # Remove x label as it's explicit enough
    p.xaxis.axis_label = None

    # set y label as 'Energy'
    p.yaxis.axis_label = "Energy [GeV]"
    p.y_range = Range1d(0, 7500)

    # Other variables
    set_extra_axes = {}
    for var_str, subdic_var in dict_var.items():
        if subdic_var["ax"] not in set_extra_axes:
            p.extra_y_ranges = {var_str: Range1d(start=subdic_var["start"], end=subdic_var["end"])}
            p.add_layout(LinearAxis(y_range_name=var_str, axis_label=var_str.capitalize()), "right")

    # List of renderers for tooltips and links
    l_r_link = []
    l_r_others = []

    # Loop over fills and plot
    for fill in dict_fills.keys():
        print("Now plotting ", dict_fills[fill]["fill"])

        # Plot energy on first y-axis
        s = p.scatter(
            x="index",
            y="LHC.BCCM.B1.A:BEAM_ENERGY",
            source=dict_fills[fill]["df"],
            line_color=color_energy_even if dict_fills[fill]["fill"] % 2 else color_energy_odd,
            alpha=0.0,
        )
        l_r_link.append(s)
        l_r_others.append(s)

        # Link points together (no renderer)
        p.segment(
            x0=dict_fills[fill]["df"].index.values[:-1],
            y0=dict_fills[fill]["df"]["LHC.BCCM.B1.A:BEAM_ENERGY"].values[:-1],
            x1=dict_fills[fill]["df"].index.values[1:],
            y1=dict_fills[fill]["df"]["LHC.BCCM.B1.A:BEAM_ENERGY"].values[1:],
            line_width=2,
            line_color=color_energy_even if dict_fills[fill]["fill"] % 2 else color_energy_odd,
        )

        # plot other variables on second y-axis
        for var_str, subdic_var in dict_var.items():
            l_r_others.append(
                p.line(
                    x="index",
                    y=subdic_var["full_name"],
                    source=dict_fills[fill]["df"],
                    line_width=2,
                    line_color=subdic_var["color"],
                    y_range_name=var_str,
                )
            )

    # Add hover tool
    hover = HoverTool(
        renderers=l_r_others,
        tooltips=[
            ("FILL", "@fill"),
            ("start", "@start"),
            ("end", "@end"),
            ("duration", "@duration"),
            ("tags", "@tags"),
            ("comment", "@comment"),
            ("link", "@link"),
        ],
    )
    p.add_tools(hover)

    # Add link if clickable_link is True
    if clickable_link:
        # Add clickable link
        tap_cb = CustomJS(
            code="""
                var l = cb_data.source.data['link'][cb_data.source.inspected.indices[0]]
                window.open(l)
                """
        )

        tapt = TapTool(renderers=l_r_link, callback=tap_cb, behavior="inspect")
        p.add_tools(tapt)

    # convert t_start to unixtime in nanoseconds
    t_start = pd.Timestamp(start_time).timestamp() * 1e9
    t_stop = pd.Timestamp(end_time).timestamp() * 1e9

    # Set x-axis range
    p.xaxis.ticker = FixedTicker(
        ticks=pd.date_range(t_start, t_stop, freq="1d").astype(int) / 10**6
    )

    if save_html:
        output_file(filename=html_filepath, title="HTML plot")
        save(p)

    return p
