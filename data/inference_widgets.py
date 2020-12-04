import numpy as np
import pandas as pd
from pathlib import Path
import os

import ipywidgets as widgets
from ipywidgets import Layout, Tab, Button, Label, Box, VBox, HBox, HTML, Output, Dropdown, BoundedIntText, Textarea, IntSlider, jslink, SelectMultiple
from IPython.display import display

dropdown_layout = Layout(width="150px")


def highlight_columns(column):
    if not np.isnan(column.Actual):
        color = 'lightgreen' if column.Actual == column.Predictions else 'orange'
    else:
        color = 'lightblue'
    bg_color = f'background-color: {color}' if column.Predictions == 1 else ''

    return [bg_color, '', '' if np.isnan(column.Actual) else bg_color]


class CroInferenceViewer():
    """Filters and displays classified paragraphs"""

    def __init__(self, df, label_list):

        # Initialize state
        self.df = df
        self.df_filtered = df
        self.label_list = label_list

        # Initialize buttons
        self.int_input = BoundedIntText(
            min=1, max=len(self.df), step=1, description="Index:")
        self.int_slider = IntSlider(
            readout=False, min=1, max=len(self.df))
        self.index_link = jslink(
            (self.int_input, 'value'), (self.int_slider, 'value'))

        # Initialize labels filter
        self.cro_cat_actual_filter = SelectMultiple(
            layout=dropdown_layout,
            options=self.label_list + ["-"],
            description="Actual Label",
            # value=label_list
        )

        self.cro_cat_predicted_filter = SelectMultiple(
            layout=dropdown_layout,
            options=self.label_list + ["-"],
            description="Predicted Label",
            # value=label_list
        )

        self.adjunct_filter = SelectMultiple(
            layout=dropdown_layout,
            options=[True, False],
            description="Adjunct pages",
            # value=label_list
        )

        # Initialize report filters
        dataset_filters = self.df.labelling_dataset.dropna().unique()
        self.dataset_filter = SelectMultiple(
            layout=dropdown_layout,
            options=dataset_filters.tolist() + ["-"],
            description="Dataset",
        )

        industries = sorted(self.df.icb_industry.unique())
        self.industries_filter = SelectMultiple(
            options=industries,
            description='Industries',
            # value=industries
        )
        countries = sorted(self.df.country.unique())
        self.country_filter = SelectMultiple(
            layout=dropdown_layout,
            options=countries,
            description="Countries",
            # value=countries
        )
        years = sorted(self.df.year.unique())
        self.year_filter = SelectMultiple(
            layout=dropdown_layout,
            options=years,
            description="Years",
            # value=countries
        )

        # Set observers
        self.int_input.observe(self.on_idx_change, 'value')
        self.cro_cat_actual_filter.observe(self.handle_filter_change, 'value')
        self.cro_cat_predicted_filter.observe(
            self.handle_filter_change, 'value')
        self.adjunct_filter.observe(self.handle_filter_change, 'value')
        self.dataset_filter.observe(self.handle_filter_change, 'value')
        self.industries_filter.observe(self.handle_filter_change, 'value')
        self.country_filter.observe(self.handle_filter_change, 'value')
        self.year_filter.observe(self.handle_filter_change, 'value')

        # Render
        self.filter_output = Output()
        self.summary_output = Output()
        self.paragraph_output = Output()
        self.render()

    def on_idx_change(self, event):
        self.render()

    def handle_filter_change(self, event):
        active_filters = []
        if len(self.cro_cat_actual_filter.value) > 0:
            only_negative = True if "-" in self.cro_cat_actual_filter.value else False
            if only_negative:
                active_filters.append(
                    "(" + " & ".join([f' {c}_actual == 0 ' for c in self.label_list]
                                     ) + ")")
            else:
                active_filters.append("(" + " | ".join(
                    [f' {c}_actual == 1 ' for c in self.cro_cat_actual_filter.value if c != "-"]
                ) + ")")

        if len(self.cro_cat_predicted_filter.value) > 0:
            only_negative = True if "-" in self.cro_cat_predicted_filter.value else False
            if only_negative:
                active_filters.append(
                    "(" + " & ".join([f' {c}_predicted == 0 ' for c in self.label_list]
                                     ) + ")")
            else:
                active_filters.append("(" + " | ".join(
                    [f' {c}_predicted == 1 ' for c in self.cro_cat_predicted_filter.value if c != "-"]
                ) + ")")

        if len(self.dataset_filter.value) > 0:
            include_nan = True if "-" in self.dataset_filter.value else False
            if include_nan:
                active_filters.append(
                    "(" +
                    f"labelling_dataset == {[v for v in self.dataset_filter.value if v != '-']}" +
                    f" or labelling_dataset != labelling_dataset"
                        + ")"
                )
            else:
                active_filters.append(
                    f"labelling_dataset == {self.dataset_filter.value}")

        if len(self.industries_filter.value) > 0:
            active_filters.append(
                f"icb_industry == {self.industries_filter.value}")

        if len(self.country_filter.value) > 0:
            active_filters.append(f"country == {self.country_filter.value}")

        if len(self.year_filter.value) > 0:
            active_filters.append(f"year == {self.year_filter.value}")

        print(active_filters)
        self.df_filtered = self.df.query(" & ".join(active_filters))

        # Reset idx value
        self.int_input.value = 1
        self.int_input.max = len(self.df_filtered)
        self.int_slider.max = len(self.df_filtered)

        self.render()

    def render_filters(self):
        self.filter_output.clear_output()
        with self.filter_output:
            display(VBox([
                HBox([
                    self.cro_cat_actual_filter,
                    self.cro_cat_predicted_filter,
                    self.adjunct_filter,
                ]),
                HBox([
                    self.dataset_filter,
                    self.industries_filter,
                    self.country_filter,
                    self.year_filter
                ]),
            ], layout=Layout(border="outset", padding="10px")))

    def render_summary(self):
        self.summary_output.clear_output()

        with self.summary_output:
            display(
                self.df_filtered[[
                    l + "_predicted" for l in self.label_list]].describe()
            )

    def render_paragraph(self):
        try:
            selected_row = self.df_filtered.iloc[self.int_input.value - 1]
        except IndexError:
            self.paragraph_output.clear_output()
            with self.paragraph_output:
                print("\n")
                print("No rows found for filter")
                print("\n")
                return

        df_result = pd.DataFrame([
            selected_row[[l + "_predicted" for l in self.label_list]].to_numpy(),
            selected_row[[l + "_prob" for l in self.label_list]].to_numpy(),
            selected_row[[l + "_actual" for l in self.label_list]].to_numpy()
        ],
            columns=self.label_list, index=[
                "Predictions", "Probability", "Actual"]
        )

        self.paragraph_output.clear_output()
        with self.paragraph_output:
            display(HBox([
                    self.int_input,
                    self.int_slider,
                    Label(f"Max: {len(self.df_filtered)}")
                    ]))

            print(
                f"\n------------------{selected_row.report_id} | p.{selected_row.page_no} ---------------------------\n")
            print(selected_row.text)
            print(
                "\n--------------------------------------------------------------------------\n")
            display(df_result.style.apply(highlight_columns))

    def render(self):
        self.render_filters()
        # self.render_summary()
        self.render_paragraph()

        display(self.filter_output)
        display(self.summary_output)
        display(self.paragraph_output)
