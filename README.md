# Adversarial-Domain-Diffuser

A domain-specific agentic framework that uses Adversarial Question Generation to identify comprehension gaps and Online Multi-Agent Diffusion Policies to coordinate complex, interpretative reasoning tasks.

## Project Structure

- `src/`: Core logic and agent implementations.
- `tests/`: Unit and integration tests.
- `data/`: Directory for synthetic samples and datasets.

## Usage

The project provides a comprehensive CLI for running the adversarial loop and evaluating performance across domains.

### Running the Adversarial Loop

To run a basic adversarial loop with default settings:
```bash
python -m src.main run --query "How do we unify physics?" --iterations 3 --visualize
```

Options:
- `--query`: The initial problem statement or question.
- `--expert-reference`: The target expert perspective.
- `--iterations`: Number of refinement iterations (default: 3).
- `--visualize`: Display a terminal-based progress chart.
- `--plot-output`: Save a visualization plot (e.g., `results.png`).

### Running Domain Evaluation

To run benchmarks across specific domains (e.g., LegalBench, MedicalQA):
```bash
python -m src.main eval --domain MedicalQA --visualize
```

Options:
- `--domain`: Specific domain to evaluate (runs all if omitted).
- `--iterations`: Iterations per loop in the evaluation.
- `--visualize`: Display summary tables and charts.
- `--plot-output`: Save evaluation summary plots.

## Visualization

The framework supports real-time progress tracking using `rich` and post-run visualizations:
- **Terminal Charts**: ASCII-based gap-closing visualization.
- **Summary Tables**: Formatted results for agents and evaluation metrics.
- **Exportable Plots**: Matplotlib-based charts for research reporting.
