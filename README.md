# TyreMind AI

TyreMind monitors a simulated truck fleet for tyre pressure, temperature, load and wear. A monitoring worker scores each tyre. High or Critical readings can open investigations. Three LLM agents use the saved measurements, model results and manufacturer manual passages to draft a health assessment, possible causes and a maintenance plan. A fleet manager reviews eligible work orders in a local Streamlit dashboard.

**Already ran TyreMind?** From the cloned project folder containing `pyproject.toml`, use the Python environment where you installed TyreMind:

```powershell
git pull
python -m streamlit run .\src\tyremind\phase4\dashboard.py
```

Open the **Local URL** printed in the terminal (usually http://localhost:8501). This reads your existing SQLite database. For a new clone, build the local data first using the steps below.

## 1. Set up a new clone

These commands use Windows PowerShell, Git and Python 3.12. Run them one at a time. Keep the repository root as your working directory so the commands share the same data and model paths.

```powershell
git clone https://github.com/Yash-15-02/TyreMind.git
cd TyreMind
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[ml,rag,agents,llm,dashboard]"
python -m pip check
```

On macOS or Linux, activate with `source .venv/bin/activate`. The first installation and model downloads require internet access. Generated data, fitted models and the local vector index are created on your machine.

## 2. Build the fleet data, models and manual index

Run these once for the default fleet of 20 trucks and 120 tyres. The generator creates 180 days of synthetic history. The training commands use time- and truck-based splits. Random Forest forecasts tread depth after another 5,000 km and Isolation Forest scores unusual operating patterns.

```powershell
tyremind-generate
tyremind-preprocess
tyremind-load
tyremind-features
tyremind-split
tyremind-train-wear
tyremind-train-anomaly
tyremind-finalize-models
```

Build the reference library for investigations. These commands download the enabled manuals, select and split passages, embed them and save them in local ChromaDB. Page references travel with each passage.

```powershell
tyremind-collect-docs
tyremind-chunk-docs
tyremind-embed-docs
tyremind-store-docs
```

## 3. Run monitoring and open the dashboard

```powershell
tyremind-worker --run-once
python -m streamlit run src/tyremind/phase4/dashboard.py
```

One cycle adds simulated readings, validates them, updates features, runs the saved models and assigns risk levels. Open **Fleet monitoring** to select a truck and tyre, view trends and inspect the rules behind its risk level.

For continuous updates, run `tyremind-worker` in a second terminal from the same repository and activated environment. It schedules a cycle every 60 seconds. Use **Refresh from SQLite** in the dashboard to see the latest saved data. Ctrl+C stops the process in the selected terminal.

## 4. Investigate an alert with local Ollama

Install and start [Ollama](https://ollama.com/download). In an activated project terminal, download the model once and set the LiteLLM model for this shell:

```powershell
ollama pull llama3.2:3b
$env:TYREMIND_LLM_MODEL = "ollama_chat/llama3.2:3b"
```

On macOS or Linux use `export TYREMIND_LLM_MODEL="ollama_chat/llama3.2:3b"`. This local setup requires no provider API key.

```powershell
python -m tyremind.agents.validate
```

This command runs the investigation through validation, reusing any saved steps. It receives an actionable event, routes it through the LangGraph supervisor, loads the tyre history and ML evidence, and runs the health, root-cause and maintenance agents with retrieval where needed. Local generation can take several minutes. If the result is `NO_WORK`, run another monitoring cycle. If it is `AWAITING_DATA_REVIEW`, the supervisor has flagged the evidence for review.

When validation returns an **investigation_id** and a validation decision, paste that ID at the prompt below:

```powershell
$investigationId = Read-Host "Investigation ID from validation"
python -m tyremind.agents.finalize $investigationId
```

Finalization publishes the result to the Phase 4 inbox. A passing draft becomes available for manager review. A failed draft appears with its validation findings and the work order is withheld. Both are valid workflow outcomes.

## 5. Review work and evaluate results

| Dashboard tab | What you can do |
|---|---|
| Fleet monitoring | Explore tyre measurements, risk history, model results and cited findings. |
| Investigation inbox | Review a published investigation. Approve an eligible draft or reject it with a reason. |
| Technician work orders | Select a manager-approved order, record actual findings and authorized maintenance, then mark completion or follow-up. |
| Evaluation | Evaluate alerts against simulated faults, compare mature forecasts with simulated tread, and save improvement proposals based on the results and feedback. |

You can also inspect results in Terminal:

```powershell
python -m tyremind.phase4.read --tyre-id TYR-0047
python -m tyremind.phase4.history
python -m tyremind.phase4.evaluate
python -m tyremind.phase4.improvements
```

The operational database is `data/tyremind.db`. The simulator keeps its own state in `data/simulator/simulator.db`. The manual index lives in `data/knowledge/chroma/`, and Phase 4 inbox exports are under `data/phase4/inbox/`. Evaluation uses synthetic outcomes. Forecast errors become available when enough later mileage has accumulated.

## Tests and design notes

```powershell
python -m unittest discover -s tests -v
```

See [simulation assumptions](docs/simulation.md), [model training results](docs/wear_model_results.md), [anomaly detection](docs/isolation_forest.md), [risk rules](docs/risk_rules.md), [document indexing](docs/chromadb.md) and [investigation routing](docs/supervisor.md).
