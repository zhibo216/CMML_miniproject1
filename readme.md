Key Code Modifications

1. Migration Direction Reversal
EC migration was corrected from downstream to strictly upstream (against flow), with polarity vectors updated accordingly. This is the most fundamental biological correction to the original codebase.

2. Periodic Boundary Conditions
Cells exiting the inlet are now rerouted back into the draining outlet, conserving total cell number — a mechanism entirely absent in the original code.

3. Bifurcation Decision Node Repositioned
The stochastic branching decision was moved from the flow-divergent node (Node 5) to the flow-convergent node (Node 15), correctly reflecting where migrating cells must choose between upstream paths.

4. Three Bifurcation Rules Implemented
BR1 (deterministic max-𝜏 τ), BR3 (equal probability), and BR5 (combined rule via 𝛼 α) were formally implemented, replacing the original single rudimentary branching rule.

5. Cell State Representation Simplified
The nested dictionary structure (num, polarity, migration) was replaced by a single NumPy array Ncell, with migration and polarity logic consolidated into migrate() and update_polarity(), eliminating the dependency on separate modules cell_migration.py and realign_polarity.py.

6. Vectorised Computation
Element-wise for loops in conductance and migration calculations were replaced with NumPy slice operations, improving both efficiency and readability.
