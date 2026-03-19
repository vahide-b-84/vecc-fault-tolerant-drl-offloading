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

### SUMO Setup
This project requires SUMO for vehicular mobility simulation.

- Install SUMO from the official website: https://www.eclipse.org/sumo/
- Ensure that SUMO is added to your system PATH
- Make sure TraCI is properly configured for Python interaction

## Running

### Step 1: Configuration
Before running the simulation, ensure that the configuration files (e.g., `configuration.py`, `params.py`) are set according to the desired scenario.

### Step 2: Pre-Simulation Setup
Run the following script to generate the required input data:

```bash
python before_Simulation.py
```

This step:
- loads the SUMO network and mobility data,
- generates RSU-to-RSU link properties,
- creates `graph_data.json` and `taskQueue.json`,
- generates parameter files:
  - `homogeneous_server_info.xlsx`
  - `heterogeneous_server_info.xlsx`
  - `task_parameters.xlsx`

This step should typically be executed only once before running all compared methods to ensure fair evaluation under identical data.

### Step 3: Main Simulation

```bash
python project_main.py
```

During simulation, raw results are saved via `save.py` in:
- `heterogeneous_results/`
- `homogeneous_results/`

### Notes
- The `before_Simulation.py` script must be executed before running the main simulation.
- Ensure that SUMO and TraCI are properly installed and configured.
- The generated input files will be overwritten if the pre-simulation step is executed again.

## Result Aggregation
After the raw simulation outputs have been generated, execute:

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
