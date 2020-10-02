# Data description

This document should give an overview on the proposed data pipeline, labelling process and training/testing/inference approach for the goal of automated identification and classification of climate-related risks and opportunities (CRO).

## Initial firm annual report dataset

The annual reports (or the national registration document if legally required) of the companies composing the STOXX Europe 50 Index (as of 24. April 2020, last review date 2. September 2019) are collected with the help of a manually assisted scraper directly from the investors relation section of each companies website and for each available year from 1999-2019.

The PDF reports then are processed and parsed with the [Apache Tika](https://tika.apache.org/) toolkit by
by extracting and storing page by page the content of the recognized text boxes in an array of pages. If available, the table of contents (ToC) which are usually stored as bookmarks in the PDF file, is also extracted. Since these reports are ususually hundreds of pages long, this information would allow for some kind of report navigation and yield more insights for downstream tasks. However, about half of the reports are missing this ToC and a some more advanced extraction process would be required to obtain the ToC.

(Optionally: The illustrations and tables via [Tabula](https://tabula.technology/) could also get extracted and stored for each page)

For each resulting company folder, the raw `*.pdf` report files and extracted `*.yml` files then are stored as a new entry in the `Firm_AnnualReport.csv` dataset. Using the company name, report type and year as id (e.g.: `es_telefonica-AR_2016`), this dataset then is merged with the `Firm_Metadata.csv` dataset that holds firm level variables such as the [ICB industry and supersector](https://www.ftserussell.com/files/support-document/icb-structure-definitions), `country` and others.

## Text preprocessing

While the PDF parsing job usually does a good job in recognizing the text layout columns and sections/paragraphs, there are usually always parsing artifacts from headers, tables, illustrations that clutter the parsed text files.

Meanwhile, climate-related risk disclosures are usually written in verbatim and often the necessary context spans multiple sentences. Consequently, the labelling, training, testing and inference steps described as below thus all rely on paragraphs as the `document` entity.

Parsed raw text of a page in question thus in all cases first needs to get preprocessed by filtering, formatting and splitting into reproducible and meaningful paragraphs. Using a rule-based approach, the class `DocumentPreprocessor` contains multiple `Regex` operations as well as some other heuristic based methods that attempt to achieve this.

## Labelling

For both rule-based and machine learning approaches, at least some labelled documents are necessary to calculate validation metrics. Thus in a first step, a certain amount of reports is labelled based on the requirements and process outlined below. To ensure labelling/coding reliability and ensure no bias, at least some of the reports should get labelled by at least two coders.

### Labelling sample selection

An initial selection of (PROPOSAL:) `n=200` reports from the `Firm_AnnualReport.csv` dataset should yield a first dataset to work with. To enable robustness checks and get insights on generalization capatabilities of the different identification and classification approaches, the (partially random) sample selection process is defined as follows:

    -   Select all firm's 2019 report (50 in total)
    -   Select two firms (PROPOSAL:) `gb_unilever_plc` ( 20 in total, 19 without 2019)
    -   The remaining 131 reports are drawn randomly, stratified by:
        -   Year --> Since the dataset is not balanced along the temporal axis
        -   Industry

These selected reports are marked accordingly in the `Firm_AnnualReport.csv` dataset by getting a `should_label` boolean flag set to `True`.

### Labelling process

The labelling process itself is configured in a [Jupyter Lab Notebook](./notebooks/Labelling.ipynb) , see the [README](./README.md) for how to get the environment setup. This notebook also contains a codebook with instructions for the labelling task. There, the `Firm_AnnualReport.csv` is opened and the `Firm_AnnualReport_Labels.pkl` dataset is initialised (or loaded from disk if already available).

Then, for each report that is flagged as `should_label`, the following procedure occurs:

1.  To speed up the process, the raw texts for each page are loaded, lemmatized and filtered by a [keyword list](./data/keyword_vocabulary.txt) of _words_ and _n-grams/phrases_ that are related to CRO's. The resulting _keyword hit table_ is displayed.

2.  Navigating through these relevant pages (the hit table), on each page change the described document preprocessing step runs and in a custom Jupyter Widget the paragraphs are displayed. The paragraph containing a hit will be highlighted. Should no paragraph contain a keyword, this should get double checked since the preprocessing step in this case may have removed the line in question.

3.  The coder goes through each paragraph and selects paragraphs that fullfill the conditions defined in the codebook. Selecting a paragraph displays the labelling fields containing:

    - `CRO_CAT`: Physical risk (**PR**), transition risk (**TR**) or opportunity (**OP**) or `None`
    - `CRO_SUB_CAT`: TCFD subcategories (see Codebook)
    - `COMMENT`: A comment field for additional information

and a new row is added in the `Firm_AnnualReport_Labels.pkl` dataset that includes the paragraph number and the document/text of the paragraph itself. Note: A paragraph should only be labelled once, i.e. no multi-class!

4.  The coder should also check each neighboring pages for relevant content and proceed as in 3.

5.  Upon finishing all pages, the button `Mark as labelled` is pressed which sets the flag `is_labelled` in `Firm_AnnualReport.csv` as `True` and the next report is displayed.

### Label postprocessing

To maximize the output of the labelling effort, each paragraph from _hit pages_ (+/- 1 neighboring) that is **NOT** labelled in the previous step as one of the main CRO categories (`CRO_CAT`) will get included in the `Firm_AnnualReport_Labels.pkl` as _negative example_.

As defined in the codebook, paragraphs with `CRO_CAT` as set as `None` are those excerpts that mention one of the key themes (_climate change_, _carbon emissions_, ...) but do not fullfill the requirements of a positive label of a CRO. Since their context is rather close to the context of positive examples, theses negative examples are considered _high quality_ and will be marked as such.

### Data Augmentation

Since the labelled sample dataset will most likely be rather small, data augmentation by replacing words/phrases with synonyms or by word embeddings could increase the amount of training data and hopefully improve the performance of the models.

The **positive** labels and, depending on the balance of the dataset, the _high quality_ **negative** examples are selected for this.

### Training

First all the labels from `Firm_AnnualReport_Labels.pkl` in reports that are

- **Not** from `2019`
- **Not** from the _held-out industry_ (PROPOSAL:) `Basic Materials`
- **Not** from the held-out company

get selected. Should this selection lead to a training dataset that exceeds `80%` of the overall labelled dataset size, the excess is deselected. This initial selection then gets partioned into a `training.pkl` and `validation.pkl` dataset (PROPOSAL: 80%/20%) while the remainder is set as `test.pkl` dataset.

The `training.pkl` and `validation.pkl` dataset then is used to select the best performing models (rule based, machine learning) and to do hyperparamater optimization.

TODO: Proposal of models:

- TASK: Multi-class vs. binary for each main category `CRO_CAT`. Multi-class of `CRO_CAT` for each main category.
- INPUT: TF-IDF (BoW), Word embeddings, BERT (contextualized word embeddings)
- CLASSIFIER: Linear, random forest, XGBoost,...

### Test

The `test.pkl` dataset containing the held out data is used for getting unbiased estimates of the out of sample performance (generalization capatabilities).

## Inference

The remaining, unlabelled reports then get processed in the following order:

1.  Keyword/n-gram search through all pages
2.  Hit pages and their immediate neighbors get selected
3.  For each selected page, run the preprocessing to partition the text into the _paragraph_ documents.
4.  Each paragraph then goes through the best in class classifier and based on the probability threshold, set accordingly.
