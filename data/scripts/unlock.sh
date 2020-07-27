#!/bin/bash
find . -iname '*.pdf' -exec qpdf --decrypt --replace-input {}  \;