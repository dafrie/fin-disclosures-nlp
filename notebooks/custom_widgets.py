from ipywidgets import Layout, Button, Label, Box, VBox, HBox, HTML


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
