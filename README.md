# Setup and run

Run these commands from the root of your cloned repository.

## 1. Install Python dependencies

Use Python 3.13, which this project was tested with. [Download Python](https://www.python.org/downloads/)

**macOS or Linux**

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

**Windows PowerShell**

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

## 2. Start Ollama and download Llama

Install Ollama and make sure its local service is running. Open the Ollama app or, if you installed the CLI only, run `ollama serve` in a separate terminal. [Download Ollama](https://ollama.com/download)

```bash
ollama pull llama3.1
```

The model download is about 4.9 GB. [Llama 3.1 model](https://ollama.com/library/llama3.1)

## 3. Build the index

```bash
python -m medical_rag.index
```

The first run downloads the MACCROBAT dataset and MiniLM from Hugging Face. Expect `Indexed 629 chunks in medical_rag_index.` Chroma saves the index in that folder, so you can reuse it the next time you open chat.

## 4. Run the chat

```bash
python -m medical_rag.chat
```

Try asking a question,  Type `/exit` to quit.

If chat cannot connect to Ollama, check that the service is running. If it cannot find the index, run Step 3 from the same directory.

