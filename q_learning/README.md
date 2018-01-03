# Description

This repository contains all files related to q-learning and backups of the files from different runs.
The q-learning code is based on code from <https://github.com/tomrunia/DeepReinforcementLearning-Atari>, with most of the files having been changed to adapt to minesweeper and with some improvements such as double q-learning etc.

# Folder descriptions

* backup_output_net1_discount_0_batch_400_best: This one got the highest overall win-rate on 6x6 boards.
* backup_output_net1_discount_0_batch_400_random_mines: The same configuration as above but trained on 1-12 mines.
* backup_output_net2_discount_0_99_batch_32_figure: Used for the figures in the report.
* backup_output_net2_discount_0_batch_32_figure: Used for the figures in the report, showing the results when the discount factor is 0.
* backup_output_net2_discount_0_batch_400_best: The best model using the second network. It seems that a high batch-size is necesssary to reach a high win-rate, otherwise it stops at some point getting much better.
* output_best: Contains the two best models from "backup_output_net1_discount_0_batch_400_best"