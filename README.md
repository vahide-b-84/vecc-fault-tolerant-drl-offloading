# VECC Fault-Tolerant DRL Offloading Framework

## Description
This repository provides a reproducible implementation of a fault-tolerant task offloading framework based on Deep Reinforcement Learning (DRL) in Vehicular Edge-Cloud Computing (VECC).

It includes:
- Proposed bi-level DRL method
- Baseline methods
- SUMO + SimPy simulation environment
- Input datasets and configuration files

## Installation
```bash
git clone https://github.com/vahide-b-84/vecc-fault-tolerant-drl-offloading.git
cd vecc-fault-tolerant-drl-offloading
pip install -r requirements.txt
```

## Running
```bash
python before_Simulation.py
python project_main.py
```

After running the simulation, the raw experiment outputs are automatically saved through `save.py` in the following directories:
- `heterogeneous_results/`
- `homogeneous_results/`

## Result Aggregation
After the simulation outputs have been generated, execute:

```bash
python final_RESULT.py
```

This script processes the previously saved raw results, adds aggregated summaries, and generates supplementary visualization outputs.

## Reproducibility
This repository includes all components required to reproduce the experimental results:
- Source code
- Baselines
- Configuration files
- Input datasets

## Citation
If you use this code, please cite the corresponding paper (currently under review):

"A Bi-Level Mobility-Aware Deep Reinforcement Learning Approach for Fault-Tolerant Task Offloading in Vehicular Edge-Cloud Computing."

Full citation details will be updated upon publication.

## License
MIT License
