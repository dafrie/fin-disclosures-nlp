import pandas as pd
import numpy as np
from IPython.display import display
from ipywidgets import Layout, Button, Label, Box, VBox, HBox, HTML, Output, Dropdown, BoundedIntText, Textarea


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
dropdown_layout = Layout(width="80px")

cro_options = [('None', ''), ('Physical Risk', 'PR'),
               ('Transition Risk', 'TR'), ('Opportunity', 'OP')]
cro_sub_type_options = [('None', ''), ('Accute', 'ACUTE'), ('Chronic', 'CHRON'),
                        ('Policy and Legal', 'POLICY'), ('Technology',
                                                         'TECH'), ('Market', 'MARKET'), ('Reputation', 'REPUT'),
                        ('Resource Efficiency', 'EFFI'), ('Energy Source', 'ENERGY'), ('Products and Services', 'PRODUCTS'), ('Markets', 'MARKETS'), ('Resilience', 'RESILI')]

label_options = [('None', ''), ('Positive', True), ('Negative', False)]


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
    comment = Textarea(
        value=row.comment if pd.notna(row.comment) else '',
        placeholder='Comment',
        disabled=False
    )
    delete_button = Button(description="Delete", button_style='danger')
    result = HBox([
        cro_type,
        cro_sub_type,
        page,
        paragraph_no,
        label,
        comment,
        delete_button
    ])
    cro_type.observe(lambda event: update_labels(
        'update', row, 'cro', event.new), names='value')
    cro_sub_type.observe(lambda event: update_labels(
        'update', row, 'cro_sub_type', event.new), names='value')
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
    items = [get_line(row, update_labels) for row in labels.itertuples()]
    grid = VBox(items)
    return grid
