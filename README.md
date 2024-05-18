Builds based on https://github.com/hdmmblz/multi-graph/tree/master

## Install

Run `pip install -r requirements.txt` to install all dependencies.

## Generate synthetic graphs

Download datasets from https://drive.google.com/file/d/1t8oKegE79ctV5aiMKyklmXbV5-fCODgk/view?usp=sharing, then unzip it under `src/`

Run `generate/generate.cpp` for synthetic graphs and `generate/generate_trans.cpp` for their transformations.

## Replicate results

Create directory `src/logging/results`.
From `src/` directory, run command `python main.py [dataset] [time_range] [num_relation]`.
Models will automatically run on static multi-relational equivalents of generated graphs.
Results will be printed to console and logged in `src/logging/results`. 
A single file will collect the last epoch for each experiment for each dataset.

A description for different datasets and the specific arguments required are as follows:

```
python src/main.py tp1 2 1                          #\varphi_1
python src/main.py tp2 2 1                          #\varphi_2
python src/main.py tp3 2 1                          #\varphi_3
python src/main.py tp1_trans 1 2                    #transformed \varphi_1
python src/main.py tp2_trans 1 2                    #transformed \varphi_2
python src/main.py tp3_trans 1 2                    #transformed \varphi_3
python src/main.py tp4 10 3                         #\varphi_4
python src/main.py tp4_trans 1 30                   #transformed \varphi_4
```

