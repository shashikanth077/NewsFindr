# NewsFindr: AI-Powered Personalized News Retrieval Agent

An agentic AI workflow, in a single Google Colab notebook, that retrieves recent, personalized news for a customer.

```
User email → SQL Agent (Groq + LangChain) → verified profile → LLM query expansion
          → DuckDuckGo News → de-dupe / allowlist / recency / relevance filters
          → LLM relevance review + grounded summaries → top-3 articles with source URLs
```

## Running in Google Colab

1. Open `NewsFindr_Agentic_AI_News_Retrieval.ipynb` in Colab (**File → Upload notebook**, or open it from GitHub).
2. Optional: upload the provided `customer.db` to the Colab file panel (`/content`) to include its customers.
   If no database is present, the notebook creates one with the sample customers (John, Sarah, David).
3. Add a Colab Secret named `GROQ_API_KEY` (key icon in the left sidebar) and enable notebook access.
   You can get a free key at <https://console.groq.com/keys>.
4. **Runtime → Run all.**

The notebook installs its own dependencies (`langchain` 1.x, `langchain-groq`, `langchain-community`, `ddgs`).
News results are fetched live from DuckDuckGo, so outputs differ between runs.

## Export for submission

* **.ipynb:** File → Download → Download .ipynb (after running all cells).
* **.html:** `jupyter nbconvert --to html NewsFindr_Agentic_AI_News_Retrieval.ipynb`.
  The last section of the notebook explains how to do this inside Colab.
