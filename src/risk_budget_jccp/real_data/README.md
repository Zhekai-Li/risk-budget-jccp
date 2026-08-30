# Real-Data Experiment Suite

This subpackage contains the M5 / Power / French real-data pipelines. Library
code lives under `src/risk_budget_jccp/real_data`, runners under
`scripts/real_data`, configs under `configs/real_data`, and generated artifacts
under the gitignored `data/*/real_data` and `artifacts/<run>/real_data` trees.

The files use compact names such as `*_train.csv` and `*_test.csv`, but their
statistical role is calibration and held-out evaluation. The calibration split
is used to construct empirical CVaR constraints, moment estimates, and
residual-flow scenarios. Held-out observations are used only for empirical
joint-violation and scalar violation-count metrics; they are not used to fit a
predictive model.

Generated solver artifacts live under `artifacts/<run>/real_data/runs/<case>`. This is
not a third evaluation split; it is the technical record of the optimization
run, including solution JSON files, solver logs, diagnostics, and case-level
tables that contain both calibration and held-out metrics. Paper-facing
evaluation outputs are organized only as `artifacts/<run>/real_data/in_sample/<case>`
and `artifacts/<run>/real_data/out_of_sample/<case>`. The public `results/`
tree retains only lightweight summary baselines; complete run details are in
the tagged GitHub Release asset.

CVaR and Bernstein optimized allocations use the requested DCA formulation
where the implemented model is genuinely coupled. Separable cases are marked as
separable equivalents instead of being presented as coupled DCA. Cantelli
optimized allocations use the documented nonlinear fallback and explicit budget
validation. The Power case expands constraints by representative snapshot,
selected branch, and flow direction, then evaluates all selected-snapshot
constraints over the held-out residual scenario set.
