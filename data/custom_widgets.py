from pprint import pprint
from ipywidgets import Layout, Tab, Button, Label, Box, VBox, HBox, HTML, Output, Dropdown, BoundedIntText, Textarea
from IPython.display import display
import numpy as np
import pandas as pd
from pathlib import Path
import os

import data.dataframe_preparation as preparation
import data.preprocessing as preprocessing
# from .. import data
# from data import dataframe_preparation as preparation
# from data import preprocessing


def render_text(text, include_line_no=True):
    """Renders a large text field with line numbers"""

    style = '' if not include_line_no else """
    <style>
        pre { counter-reset: line;}
        code { counter-increment: line; }
        code:before { display: inline-block; content: counter(line); width: 40px; background-color: #E8E8E8;}
    </style>
    """
    lines = [str(f"<code>{line}</code>\n") for line in text.splitlines()]
    content = f"<pre>\n{''.join(lines)}</pre>"
    widget = HTML(value=style+content)
    return widget


int_layout = Layout(width="40px")
dropdown_layout = Layout(width="120px")
text_layout = Layout(width="500px")
comment_layout = Layout(width="300px")

cro_options = [('None', ''), ('Physical Risk', 'PR'),
               ('Transition Risk', 'TR'), ('Opportunity', 'OP')]
cro_sub_type_options = [('None', ''), ('PR - Accute', 'ACUTE'), ('PR - Chronic', 'CHRON'),
                        ('TR - Policy and Legal', 'POLICY'), ('TR - Technology',
                                                              'TECH'), ('TR - Market', 'MARKET'), ('TR - Reputation', 'REPUT'),
                        ('OP - Resource Efficiency', 'EFFI'), ('OP - Energy Source', 'ENERGY'), ('OP - Products and Services', 'PRODUCTS'), ('OP - Markets', 'MARKETS'), ('OP - Resilience', 'RESILI')]

label_options = [('None', ''), ('Positive', True), ('Negative', False)]


def handle_cro_sub_type_update(event, row, update_labels, cro_type_field):
    update_labels('update', row, 'cro_sub_type', event.new)
    if event.new:
        index = [option[1] for option in cro_sub_type_options].index(event.new)
        new_cro_value = cro_sub_type_options[index][0].split(" - ")[0]
        cro_type_field.value = new_cro_value


def get_line(row, update_labels):
    cro_type = Dropdown(value=row.cro if pd.notna(
        row.cro) else '', options=cro_options, layout=dropdown_layout)
    cro_sub_type = Dropdown(value=row.cro_sub_type if pd.notna(
        row.cro_sub_type) else '', options=cro_sub_type_options, layout=dropdown_layout)
    page = BoundedIntText(
        value=row.page if pd.notna(row.page) else '',
        min=0,
        max=999,
        step=1,
        layout=int_layout,
        disabled=True
    )
    paragraph_no = BoundedIntText(
        value=row.paragraph_no if pd.notna(row.paragraph_no) else '',
        min=0,
        max=999,
        step=1,
        layout=int_layout,
        disabled=True
    )
    label = Dropdown(value=row.label if pd.notna(
        row.label) else '', options=label_options, layout=dropdown_layout)
    text = Textarea(
        value=row.text if pd.notna(row.text) else '',
        placeholder='Text',
        disabled=True,
        layout=text_layout
    )
    comment = Textarea(
        value=row.comment if pd.notna(row.comment) else '',
        placeholder='Comment',
        disabled=False,
        layout=comment_layout
    )
    delete_button = Button(description="Delete", button_style='danger')
    result = HBox([
        cro_type,
        cro_sub_type,
        page,
        paragraph_no,
        text,
        comment,
        delete_button
    ])
    cro_type.observe(lambda event: update_labels(
        'update', row, 'cro', event.new), names='value')
    cro_sub_type.observe(lambda event: handle_cro_sub_type_update(
        event=event, row=row, update_labels=update_labels, cro_type_field=cro_type), names='value')
    page.observe(lambda event: update_labels(
        'update', row, 'page', event.new), names='value')
    paragraph_no.observe(lambda event: update_labels(
        'update', row, 'paragraph_no', event.new), names='value')
    label.observe(lambda event: update_labels(
        'update', row, 'label', event.new), names='value')
    comment.observe(lambda event: update_labels(
        'update', row, 'comment', event.new), names='value')
    delete_button.on_click(lambda event: update_labels('delete', row))
    return result


def render_label_list(labels, update_labels):
    field_descriptions = HBox([Label("CRO", layout=dropdown_layout), Label(
        "Sub type", layout=dropdown_layout), Label("Page", layout=int_layout), Label("Paragraph", layout=int_layout), Label("Text", layout=text_layout), Label("Comment", layout=comment_layout)])
    items = [get_line(row, update_labels) for row in labels.itertuples()]
    items.insert(0, field_descriptions)
    grid = VBox(items, layout=Layout(
        border="1px solid lightgrey", padding="10px 0px"))
    return grid


class ReportsLabeler():
    def __init__(self, master_input_path, label_output_fn='Firm_AnnualReport_Labels.pkl', keyword_vocabulary_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'keyword_vocabulary.txt')):
        self.master_input_path = master_input_path
        self.label_output_path = os.path.join(
            os.path.dirname(self.master_input_path), label_output_fn)
        self.df_master = pd.read_csv(self.master_input_path)
        self.df_master = self.df_master.set_index("id")
        self.load_label_file()

        self.vocabulary = preparation.get_keywords_from_file(
            os.path.join(keyword_vocabulary_path))

        # Initialize state
        self.current_report_index, self.number_unlabelled_reports = self.get_next_idx_of_report()
        self.current_report = None
        self.counts_table = None

        # Initialize buttons
        self.next_report_button = Button(
            description="Mark as labelled")
        self.next_relevant_page_button = Button(
            description="Next relevant page")
        self.prev_relevant_page_button = Button(
            description="Previous relevant page")
        self.page_index_input_field = BoundedIntText(
            value=1, description='Page No:', disabled=False, min=1, max=999)

        # Set button callbacks
        self.next_report_button.on_click(
            lambda event: self.on_next_report_button_clicked(event, 1))
        self.page_index_input_field.observe(
            lambda event: self.on_page_change(event.new), 'value')
        self.prev_relevant_page_button.on_click(
            lambda event: self.on_relevant_page_toggle('previous'))
        self.next_relevant_page_button.on_click(
            lambda event: self.on_relevant_page_toggle('next'))

        # Initialize first page
        self.render()
        self.on_next_report_button_clicked(None, 0)
        self.on_page_change(0)

    def load_label_file(self):
        output_path = Path(self.label_output_path)
        if output_path.is_file():
            self.df_labels = pd.read_pickle(output_path)
        else:
            self.df_labels = pd.DataFrame(columns=[
                                          'report_id', 'cro', 'cro_sub_type', 'page', 'paragraph_no', 'label', 'comment', 'text'])
            self.save_label_file()

    def save_master_file(self):
        self.df_master.to_csv(self.master_input_path)

    def save_label_file(self):
        self.df_labels.to_pickle(self.label_output_path)

    def get_next_idx_of_report(self):
        all_unlabelled = self.df_master.loc[(
            self.df_master['should_label']) & ~self.df_master['is_labelled']]
        index = all_unlabelled.iloc[[0, ]].index[0]
        if not index:
            return print("All reports seem to be labelled")
        return index, len(all_unlabelled)

    def update_labels(self, action, row, field_name=None, field_value=None, skip=False):
        if action == 'insert':
            self.df_labels = self.df_labels.append(row, ignore_index=True)
        elif action == 'delete':
            self.df_labels = self.df_labels.drop(row.Index)
        elif action == 'update':
            self.df_labels.at[row.Index, field_name] = field_value
        if not skip and not action == 'update' or field_name == 'page_no':
            self.update_widgets()
            self.render_labelling_output()
        if not skip:
            self.save_label_file()

    def update_widgets(self):
        self.label_list_output.clear_output()
        with self.label_list_output:
            labels = self.df_labels[self.df_labels['report_id']
                                    == self.current_report_index]
            label_list = render_label_list(
                labels, self.update_labels)
            display(label_list)

    def render_labelling_buttons(self, doc, ensure_no_duplicates=False):
        rows = []
        paragraphs = doc.split('\n\n')
        counts = preparation.get_count_matrix(paragraphs, self.vocabulary)
        page_keyword_count = counts.sum(axis=1)

        def _on_button_clicked(event, number):
            paragraph_no = int(event.description)
            paragraphs = doc.split('\n\n')
            self.update_labels('insert', {
                'report_id': self.current_report_index,
                'page': self.page_index_input_field.value,
                'paragraph_no': paragraph_no,
                'text': paragraphs[paragraph_no],
                'label': True
            })

        for idx, paragraph in enumerate(paragraphs):
            is_already_selected = False
            if ensure_no_duplicates:
                is_already_selected = True if ((self.df_labels['report_id'] == self.current_report_index) & (
                    self.df_labels['page'] == self.page_index_input_field.value) & (self.df_labels['paragraph_no'] == idx)).any() else False
            paragraph = f"<p style='background-color: {'yellow' if page_keyword_count[idx] > 0 else 'none'};'>{paragraph}</p>"
            paragraph_button = Button(
                description=str(idx), disabled=is_already_selected)
            paragraph_button.on_click(
                lambda event: _on_button_clicked(event, idx))
            rows.append(HBox(
                [paragraph_button, HTML(value=paragraph)]))
        return VBox(rows)

    def render_labelling_output(self):
        self.current_page_output.clear_output()
        with self.current_page_output:
            processed_doc = preprocessing.DocumentPreprocessor(
                self.current_report).process()

            labelling_widget = self.render_labelling_buttons(processed_doc)
            raw_doc_widget = render_text(self.current_report)
            preprocessed_doc_widget = render_text(processed_doc)

            tab = Tab()
            tab = Tab(
                [labelling_widget, raw_doc_widget, preprocessed_doc_widget])
            tab.set_title(0, "Labelling")
            tab.set_title(1, "Raw",)
            tab.set_title(2, "Processed")
            display(tab)

    def on_page_change(self, new_value):
        with self.current_page_output:
            selected_row = self.df_master.loc[self.current_report_index]
            path = selected_row['input_file']
            folder = os.path.dirname(path)
            text_file_path = os.path.join(folder, selected_row['orig_report_type'] + '_' + str(
                int(selected_row['year'])), selected_row['output_file'])
            self.page_index_input_field.value = new_value
            self.current_report = preparation.get_text_from_page(
                text_file_path, self.page_index_input_field.value)
            self.update_widgets()
            self.render_labelling_output()

    def on_relevant_page_toggle(self, direction, add_adjunct_pages=True):
        new_page = self.page_index_input_field.value
        has_current_page = bool(len(
            self.counts_table[self.counts_table.index == self.page_index_input_field.value]))
        try:
            if direction == 'next':
                new_page = self.counts_table[self.counts_table.index >
                                             self.page_index_input_field.value].iloc[0].name
            else:
                new_page = self.counts_table[self.counts_table.index <
                                             self.page_index_input_field.value].iloc[-1].name
            # Add adjunct pages (+/- 1)
            if add_adjunct_pages and abs(new_page - self.page_index_input_field.value) > 1:
                correction = 1 if direction == 'next' else -1
                new_page = self.page_index_input_field.value + \
                    correction if has_current_page else new_page - correction

        except IndexError as err:
            # Make sure that for the last keyword hit the adjunct pages are also added
            if add_adjunct_pages and has_current_page:
                if direction == 'next':
                    new_page += 1
                else:
                    new_page -= 1
            pass
        finally:
            self.page_index_input_field.value = new_page

    def on_next_report_button_clicked(self, b, direction):
        self.current_report_output.clear_output()
        self.current_page_output.clear_output()
        with self.current_report_output:
            if direction > 0:
                self.df_master.loc[self.current_report_index,
                                   'is_labelled'] = True
                self.save_master_file()
                self.save_label_file()
                self.current_report_index, self.number_unlabelled_reports = self.get_next_idx_of_report()
            selected_row = self.df_master.loc[self.current_report_index]

            path = selected_row['input_file']
            folder = os.path.dirname(path)
            text_file_path = os.path.join(folder, selected_row['orig_report_type'] + '_' + str(
                int(selected_row['year'])), selected_row['output_file'])

            # Update widgets
            display(HBox(
                (Label(f'Current report: {self.current_report_index} / Remaining: {self.number_unlabelled_reports}'), self.next_report_button)))

            # input_file = FileLink(selected_row['input_file'])
            # display(input_file)

            # If the PDF file should get opened
            # !open "$input_file"

            # Long running...
            self.counts_table = preparation.get_counts_per_page(
                text_file_path, self.vocabulary)
            print("Keyword count table:")
            pprint(self.counts_table)
            self.page_index_input_field.value = 0

    def render(self):
        self.current_report_output = Output()
        self.current_page_output = Output()
        self.label_list_output = Output()

        display(self.current_report_output)
        display(HBox((self.prev_relevant_page_button,
                      self.page_index_input_field, self.next_relevant_page_button)))
        display(self.label_list_output)
        display(self.current_page_output)
