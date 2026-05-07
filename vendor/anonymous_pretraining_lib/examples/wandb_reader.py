"""This script demonstrates how to retrieve data from wandb using the anonymous_pretraining_lib library."""

import anonymous_pretraining_lib as apl

config, df = apl.utils.reader.wandb_run(
    "excap", "single_dataset_sequential", "p67ng6bq"
)
print(df)
configs, dfs = apl.utils.reader.wandb_project("excap", "single_dataset_sequential")
print(dfs)
