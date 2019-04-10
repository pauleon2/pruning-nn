# Description of the executed experiments

### Experiment 1
iterative pruning, 4 different initial models all 28x28 - 100 - 10
- OBD
- magnitude (3x)
- random

2 retrain epochs fixed
\
percentages: 70, 60, 50, 40, 25
\
25 runs for each experiment

### Experiment 2
Info: Experiment 2, 3 and 4 share the same underlying model again 28x28 - 100 - 10 architecture and an original accuracy of 97.29 %.

- OBD
- magnitude (x3)
- random

1 model
\
25 runs with variable retraining, 25 runs with fixed retraining (2 epochs)
\
50 percent iterative pruning

### Experiment 3 (needs extension)

fixed number pruning: 10000, 5000, 1000

### Experiment 4 (needs extension)

single pruning

### Experiment 5

use of a bigger model compared to the other ones which all used an arcitecure 784 - 100-10 we use now 784-300-100-10 what increases the total weight count singificantly.

### Baseline Experiment (needs evaluation)