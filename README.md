# ADGNN

* Install all the requirements from requirements.txt.
* Please use the following command to generate the protected graph. You can use `--data_file` to set your local data to protect and use `--defense_budget` to set the number of injected guardian nodes. Also, `--depochs` denotes the epoch number to generate the protected graph.

  ```commandline
  python generate_defense_graph.py --data_file raw_graph/cora_time_t --defense_budget 0.1 --depochs 100 
  ```
