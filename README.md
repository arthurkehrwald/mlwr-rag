# Installation

1. Repository klonen `git clone https://github.com/arthurkehrwald/mlwr-rag.git`
2. Im Repository eine Datei mit dem Namen `.env` und folgendem Inhalt anlegen:
    ```
    LANGSMITH_API_KEY=?
    LANGSMITH_TRACING=true
    OPENAI_API_KEY=?
   ```
3. API Keys bei [Langchain](https://www.langchain.com) und [OpenAI](https://platform.openai.com) anlegen und in der Datei eintragen.
4. Den Python Package Manager [uv](https://docs.astral.sh/uv/getting-started/installation/) installieren
5. Wenn Python nicht schon installiert ist, den Befehl `uv python install` ausführen
6. Im Projektordner eine virtuelle Python Umgebung anlegen `python3 -m venv .venv` und aktivieren (Abhängig vom Betriebssystem. Auf Windows: `.\.venv\Scripts\activate`)
7. Im Projektordner den Befehl `uv pip install -r pyproject.toml` ausführen, um die benötigten Packages zu installieren.
8. Java installieren (Damit [OpenDataLoader](https://github.com/opendataloader-project/opendataloader-pdf) die PDF Dateien lesen kann)

Die Schritte 6 und 7 können einige IDEs (z. B. PyCharm) automatisch erledigen.

# Anmerkungen

Die Nutzung der OpenAI API kostet jedes Mal, wenn man das Notebook ausführt, ungefähr einen Cent. Der Code wurde überwiegend von folgenden Langchain Tutorials übernommen:
- [Build a RAG agent with LangChain](https://docs.langchain.com/oss/python/langchain/rag)
- [Evaluate a RAG application](https://docs.langchain.com/langsmith/evaluate-rag-tutorial)