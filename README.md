# Dataset
This study focuses on software defect prediction for Java projects.
The experimental data is sourced from the PROMISE public dataset, a well-known benchmark collection for software engineering research.

Five Java projects from the PROMISE dataset were selected for the experiments.
For within-project defect prediction, the lower version of each project is used as the training set, while the higher version is used as the test set.

Example:
If project X has versions 1.0 and 2.0 in the dataset,

Version 1.0 → Training data

Version 2.0 → Testing data

This setup simulates a realistic scenario where defect prediction is performed for an upcoming release based on historical data from earlier releases.

