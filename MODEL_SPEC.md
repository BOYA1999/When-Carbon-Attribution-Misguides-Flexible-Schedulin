# Computation contract

The study uses a transparent eight node radial DC microgrid with 48 half hour periods per day. Node 0 is the point of common coupling. Photovoltaics connect at node 5, storage at node 3, and HVAC, electric vehicle and shiftable task loads at nodes 2, 6 and 7. The complete numerical configuration is frozen in `configs/pedf8.json`.

The branch flow model represents active power, current magnitude squared and voltage magnitude squared. After each convex solve, a radial backward and forward sweep restores the physical equality between current, power and voltage. All reported costs, source emissions and constraint checks use that recovered flow.

Storage advances both energy and carbon stock. Charging adds the carbon attributed at the storage bus after conversion loss. Discharging removes carbon at the inventory intensity carried from earlier charging periods. The primary daily carbon boundary uses the first Great Britain ACI of the day. Prespecified sensitivity cases use zero carbon or continuous cross day propagation along the B0 reference trajectory. Storage begins and ends each day at 50% state of charge. Terminal storage carbon is not fixed.

The economic baseline B0 minimizes electricity and storage throughput cost. B1 schedules with Great Britain average carbon intensity. B2 uses nodal average intensity without storage memory. B3 retains storage carbon memory. P uses response based marginal carbon intensity computed by finite perturbations around B0. B4 directly minimizes reported source emissions. Every carbon scheduling method shares the same feasible set, cost cap and recovery procedure.

For each flexible node and period, MCI is estimated from central finite differences when both perturbation directions solve. Every device reoptimizes inside each perturbation, so MCI follows the economic reoptimization path. The perturbation is the larger of 0.1 kW and one percent of period load. Frozen sensitivity multipliers are 0.5, 1 and 2. The main action budget is two percent above B0 cost. The five point budget scan uses 0, 1, 2, 5 and 10 percent.

The primary effect is `100 * (E_B3 - E_P) / E_B0`, with the day as the statistical unit. The first 21 days form the primary set. The last seven days form a later window check. Circular moving block bootstrap intervals use 10,000 replicates at block lengths of two, three and four days with seed 20260802.

The signal role ablation replaces B3 action intensity with `(1 - alpha) * ACI + alpha * MCI` for alpha equal to 0, 0.25, 0.5, 0.75 and 1. It equalizes scheduling solves, not the upstream construction cost of MCI. Temporal controls use all 47 nonzero circular shifts, 24 four step block permutations and 24 cross day same slot substitutions for every target day.

Storage charge and discharge complementarity is recovered deterministically after the relaxed solve. Representative SCIP mixed integer comparisons use three dates, two budgets and B3, P and B4, giving 18 cases with a configured relative gap limit of `1e-6`. Agreement at 2% is a sampled consistency check. It is not a claim of exact recovery on every case.
