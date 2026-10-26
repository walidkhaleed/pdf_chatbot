# Free PDF RAG Chatbot — Google Colab

A small, **API-key-free PDF question-answering notebook** using a locally executed open-weight Qwen model inside Google Colab. It is a separate educational project from Chat Scholar.

**No paid inference API is required.** Public model weights are downloaded from Hugging Face, and inference runs within the Colab runtime using PyTorch. Google's free Colab resources may have availability, memory, and usage limits. You do **not** need an OpenAI key, a Groq key, or a Hugging Face token for these public models.

## Features

- Upload a text-based PDF directly in Colab.
- Extract pages with **LangChain's `PyPDFLoader`** and split overlapping passages with `RecursiveCharacterTextSplitter`.
- Create local **MiniLM sentence embeddings** and search them with **FAISS**.
- Generate document-grounded responses with **Qwen2.5 Instruct** using Hugging Face Transformers.
- Rewrite follow-up questions using short in-memory conversation history.
- Display **retrieved source IDs, page numbers, and similarity scores** beside model-generated answers.
- Explicitly instruct the model to abstain when supporting evidence is absent.

## Architecture

```text
PDF file
   |
PyPDFLoader (page text)
   |
RecursiveCharacterTextSplitter (900 characters, 150 overlap)
   |
all-MiniLM-L6-v2 -> normalized vectors
   |
FAISS IndexFlatIP (cosine-equivalent similarity)
   |
User question -> MiniLM embedding -> top-k passages
   |
Qwen2.5-Instruct running on Colab GPU / CPU
   |
Answer + [S#] citations + retrieved page/source listing
```

The workflow uses **semantic retrieval only**, not hybrid BM25 search. Cited evidence is shown for review, but the notebook **does not independently validate whether a generated claim is supported** by its cited excerpt.

## Start in Google Colab

1. Download `Free_PDF_RAG_Colab.ipynb` from this repository or open it through [Google Colab](https://colab.research.google.com/).
2. In Colab, choose **File → Upload notebook** (or select the notebook from GitHub).
3. If a free GPU is offered, choose **Runtime → Change runtime type → T4 GPU**.
4. Run the numbered notebook cells **in order**. The first cell installs dependencies.
5. Upload a **text-based PDF** when prompted. The demo limits documents to 30 pages.
6. Ask questions with `ask_pdf(...)`. Look at the source IDs and page numbers underneath the answer.

**Quick examples:**

```python
result = ask_pdf("Summarize the main argument in three sentences.")
result = ask_pdf("What evidence supports that conclusion?")
result = ask_pdf("Which limitations does the author discuss?")
```

After publishing this repository to GitHub under the suggested name `free-colab-pdf-rag`, the notebook will be available at:

`https://colab.research.google.com/github/walidkhaleed/free-colab-pdf-rag/blob/main/Free_PDF_RAG_Colab.ipynb`

## Tech stack

| Role | Technology |
| --- | --- |
| Notebook runtime | Google Colab / Python |
| PDF extraction | LangChain Community / PyPDFLoader / pypdf |
| Text splitting | `langchain-text-splitters` |
| Embeddings | Sentence Transformers, `all-MiniLM-L6-v2` |
| Vector retrieval | FAISS (`IndexFlatIP`) |
| Text generation | Qwen2.5-Instruct + Transformers + PyTorch |
| Conversational context | In-memory Python history |

On a GPU runtime, the notebook uses `Qwen/Qwen2.5-1.5B-Instruct`. On CPU it uses the smaller `Qwen/Qwen2.5-0.5B-Instruct` as a fallback. Both are public models; the initial model download requires network access.

## Why no API key?

A hosted inference API (for example OpenAI or Groq) receives your prompt over a network connection and computes the answer remotely. Here, Colab downloads the public Qwen model weights and then executes `model.generate(...)` directly in the Colab process. **Google provides the virtual machine**; that is not the same as having no compute cost or operating on your own laptop.

Uploaded PDFs and prompts remain in the Google-hosted Colab session for inference, rather than being sent to a separate hosted LLM API. Do not upload sensitive documents without permission.

## Troubleshooting

**`AttributeError: ... 'shape'` after `apply_chat_template`:** Use `return_dict=True` and slice generated outputs using `inputs["input_ids"].shape[1]`, as implemented in this notebook. The whole `BatchEncoding` object is not a tensor.

**`ModuleNotFoundError`:** Rerun the installation cell. If Colab requests a runtime restart after installation, restart and rerun in order.

**CUDA out-of-memory:** Switch to a smaller model or lower `max_new_tokens` and retrieved `top_k`. Free-tier GPU availability is not guaranteed.

**PDF contains no text:** Image-only or scanned PDFs require an OCR stage, which this notebook does not include.

**Hallucinations or wrong citations:** This prototype uses prompt-based grounding. Always inspect retrieved excerpts. Production use would require citation checking, better retrieval, safety controls, and evaluation.

## Requirements

The notebook installs dependencies directly. `requirements.txt` lists the same primary Python dependencies for reference. On a fresh local machine you will also need an appropriate PyTorch installation for your CPU/GPU; Colab ordinarily provides PyTorch.

## Limitations and project scope

- A tutorial/prototype, not a production application or secure document service.
- No OCR, BM25 hybrid retrieval, long-term conversation persistence, or rigorous semantic evidence checking.
- The model is small, so reasoning quality and instruction-following are limited.
- Notebook memory, uploaded data, and indexes are temporary.
- Usage can be free within Colab's free allocation but is **not guaranteed unlimited or always available**.

## References

- [Hugging Face chat-template documentation](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Qwen2.5-1.5B-Instruct model](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [MiniLM embeddings model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Google Colab](https://colab.research.google.com/)
