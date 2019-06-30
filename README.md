# Legal Topic Modeling (Docket Watch)

This work was done while a fellow with [Insight Data Science](https://www.insightdatascience.com/), in consultation with [Ping](https://www.timebyping.com/).

In this repository we experiment with various preprocessors to build a well-performing topic model (using [gensim's](https://radimrehurek.com/gensim/) LDA model) from public-record court rulings downloaed from [The Caselaw Access Project](https://case.law). Success is measured primarily by coherence, but we also compute perplexity, and a custom validation metric based on the network of citations among legal documents is also computed.

## Datasets referenced in this repository can be downloaded from my Dropbox [here](https://www.dropbox.com/sh/4i7vncum9q73d2z/AABEC7tLOA-7TefUYS81wHrza?dl=0). Notebooks expect input datasets to be in a folder called "data_uncompressed".

### Input datasets (/data_uncompressed)
* Every input dataset contains opinions from cases which appear in the citation network for the corpus (that is, every such case cites or is cited by at least one other case in the corpus)
* Each of the raw datasets with 12k rulings is contained in the /data_uncompressed/raw_data_12k folder.
    - random_cases2_raw.csv: contains 4000 opinions randomly sampled from each of the jurisdictions of Arkansas, Illinois, and New Mexico; no restrictions are placed on decision date
    - cases_IL_12k_raw.csv: contains 12000 opinions randomly sampled from the Illinois jurisdiction; no restrictions are placed on decision date
    - cases_after1950_raw.csv: contains 12000 opinions randomly sampled from each of the jurisdictions of Arkansas, Illinois, and New Mexico; cases must have been decided after 1950
    - cases_IL_after1950_raw.csv: contains 12000 opinions randomly sampled from the jurisdiction of Illinois which were decided after 1950
* cases_IL_after1950_42k.csv contains all opinions from the corpus which lie in the citation network, are from the Illinois jurisdiction, and which were decided after 1950. There are roughly 42000 such cases
* each preprocessor and dataset size has a folder containing the preprocessed dataset (e.g. baseline_12k)
* attorneys, judges, metadata, and opinions all contain raw data from each jurisdiction from The Caselaw Access Project; these datasets were pushed to a local PostgreSQL database
* citation_graph_full_data.csv and citation_graph_data_no_ops.csv both contain data from each case in the citation network. The full_data dataset contains opinions, while the no_ops dataset contains only metadata. See below for an explanation of the citation network

__________________________________________________________________

The repository directory is laid out as follows:

### Experimentation Notebooks

Contains several notebooks implementing various preprocessing routines, training LDA models of varying numbers of topics on the preprocessed corpora, and some short visualizations to compare performance across different input datasets.

There are also several stand-alone notebooks performing comparisons of different models, finding "legal stopwords", or analyzing the network of legal citations in the corpus.


* <b><u>Preprocessor naming convention</u></b>
    - baseline: tokenize, remove English stop words, keep only nouns, verbs, adjectives, and adverbs, and lemmatize
    - basic_stopwords: baseline + removal of "legal stopwords"
    - basic_stopwords_nouns: baseline + removal of "legal stopwords" + only nouns kept
    - basic_stopwords_nouns_verbs: baseline + removal of "legal stopwords" + only nouns and verbs kept
    - phrasing: baseline + bigram phrasing (min_count=5; threshold=100)
    - phrasing_basic_stopwords: baseline + removal of "legal stopwords" + bigram phrasing
* <b><u>Sizing convention</u></b>
    - 12k corresponds to any one of the datasets above with 12k opinions
    - 42k corresponds to the dataset with all Illinois cases decided after 1950
* <b><u> Models with high numbers of topics </u></b>
    - The preprocessors basic_stopwords and basic_stopwords_nouns_verbs each have a special topic modeling pipeline which trains and validates topic models with 25, 50, 75, and 100 topics each (up from the 5, 8, 10, 12, 15 topic models used for iteration)
* <b><u>The Citation Network</u></b>
    - Construction
        1. Using regex matching, extract any citations from the corpus which use a reporter from the corpus (e.g. "Ill., Ill.2d")
        2. Prune these citations for those referencing cases in the corpus
        3. Construct a table of pairs (citing_case_id, cited_case_id)
        4. Build a (non-directed) network whose nodes are the case id's and where edges indicate that one of the cases has cited the other.
    - citation_graph_EDA.ipynb
        * explores information about the cases involved in the largest connected component of the citation graph
    - The citation network is used to validate the topic models as follows (the metric is called *citation-similarity*:
        1. Using the topic model, obtain the topic distributions for all  documents in the corpus; think of these as vector-embeddings of the documents.
        2. Train a KNN model on the vectorized documents.
        3. Given a document, find its 5 nearest neighbors (this number can be tuned).
        4. Compute the minimum, average, and maximum edge-length in the citation network between the given document and its nearest neighbors.
* <b><u> compare_*_.ipynb </u></b>
    - These notebooks compare the perplexity, coherence, and citation-similarity scores of various topic models
* <b><u> find_legal_stopwords.ipynb </u></b>
    - This notebook details the prodedure for finding the list of legal stopwords
    
### Validation Output
This is where I store all of the validation output from the various models in the experimentation notebooks.

### Wordclouds
This is where wordclouds of all of the topics for all of the models are stored.

### best_pipeline.py
This script accepts a directory of .txt files (the input corpus of legal documents), applies the basic_stopwords_nouns_verbs preprocessor, and trains an LDA model with 50 topics. PyLDAvis can also be used to interactively visualize topics (WARNING: pyLDAvis can take quite some to process)

### citation_graph.gpickle
This is the actual citation network object. The package [networkx](https://networkx.github.io/) is required to load it.

### MVP.py
This is a script implementing the baseline model for preprocessing and topic modeling.
