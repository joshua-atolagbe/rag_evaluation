# RAG Eval

**RAG Eval** is a Python package designed for evaluating Retrieval-Augmented Generation (RAG) systems. It provides a suite of tools to score generated responses across various metrics, including query relevance, factual accuracy, coverage, coherence, and fluency.


## Features

- **Multi-Metric Evaluation:** Evaluate responses using a variety of metrics:
  - **Query Relevance**
  - **Factual Accuracy**
  - **Coverage**
  - **Coherence**
  - **Fluency**
- **Standardized Prompting:** Uses a well-defined prompt template to assess responses consistently.
- **Customizable:** Easily extendable to add new metrics or evaluation criteria.
- **Easy Integration:** Provides a high-level function to integrate evaluation into your RAG pipelines.

## Installation


### Using pip

```bash
pip install rag-eval

### Usage

import openai
from rag_eval import get_openai_key, evaluate_response

# Set your OpenAI API key either via an environment variable or directly:
api_key = get_openai_key("FALLBACK_API_KEY")
openai.api_key = api_key

query = "Could you provide an overview of the training session?"
response_text = (
    "The training session covered testing procedures and tool usage. "
    "It was designed for both new and experienced users."
)
document = "Training session document content goes here..."

**Evaluate the response using the default model ("gpt-4o-mini" or your choice)**
report = evaluate_response(query, response_text, document, model="gpt-4o-mini")
print(report)

This function returns a pandas DataFrame with:

Metric Names: Query Relevance, Factual Accuracy, Coverage, Coherence, Fluency.

Normalized Scores: A 0–1 score for each metric.

Percentage Scores: The normalized score expressed as a percentage.

Overall Accuracy: A weighted average score across all metrics.

