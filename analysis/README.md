## Analysis

Our sampling scripts produce visualizations and some simple analysis in the Weights and Biases UI.

For more in-depth exploration, we provide an analysis notebook, adapted from that of [MDGen](https://github.com/bjing2016/mdgen).

First, add the details of the sampling runs to a CSV file `SAMPLE_RUNS_CSV`, following the structure of ['sample_runs.csv'](https://github.com/prescient-design/jamun/blob/main/analysis/sample_runs.csv).

Then, precompute analysis results with:
```bash
python analysis_sweep.py --csv [SAMPLE_RUNS_CSV] --experiment [EXPERIMENT] --output-dir [ANALYSIS_OUTPUT_DIR]
```

Finally, run [`make_plots.ipynb`](https://github.com/prescient-design/jamun/blob/main/analysis/make_plots.ipynb) to make plots for your chosen experiment.

For example, if we have details for our sampling runs at `sample_runs.csv`:
```txt
experiment,wandb_sample_run_path,reference,trajectory
Our_2AA,prescient-design/jamun/sc3roq1e,JAMUNReference_2AA,JAMUN
Our_2AA,prescient-design/jamun/ur17v6od,JAMUNReference_2AA,JAMUN
Timewarp_2AA,prescient-design/jamun/jls6fodw,TimewarpReference,JAMUN
...
```

We can run the analysis for `Timewarp_2AA` with:
```bash
python analysis_sweep.py --csv sample_runs.csv --experiment Timewarp_2AA --output-dir ./jamun-analysis/
```
