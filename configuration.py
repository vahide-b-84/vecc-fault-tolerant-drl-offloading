# configuration.py
# Central configuration for the simulation + RL agents.
# Tip: Call `parameters.validate_config()` at the very beginning of params.py to avoid silent misconfigurations.
from scipy.stats import norm

class parameters:
    # ======================================================================
    # 1) High-level experiment switches
    # ======================================================================

    # Reliability profile type:
    # - "homogeneous"   : all nodes within the same class share a narrow failure-probability interval
    # - "heterogeneous" : nodes have wider variability in failure-probability interval
    SCENARIO_TYPE = "heterogeneous"  # Options: "homogeneous", "heterogeneous"

    # Which reliability state to use as the "base" condition for the run.
    # This maps to the states defined in STATES below.
    FAILURE_STATE = "high"  # Options: "low", "med", "high"

    # Which policy/algorithm to run (used for experiment selection + folder naming).
    model_summary = "dqn"  # Options: "dqn", "ppo", "greedy", "NoForwarding"

    # Architecture flag:
    # - True  : flat/unified decision-making (single-level)
    # - False : hierarchical two-level decision-making
    Flat_mode = False  # False / True

    # ======================================================================
    # 2) Data quality / robustness scenarios (mutually exclusive)
    # ======================================================================

    # Scenario selector:
    # - "base"             : clean inputs (both probabilities must be 0.0)
    # - "missing_data"     : missing data injection controlled by missing_data_p
    # - "trajectory_noise" : trajectory noise injection controlled by trajectory_noise_p
    scenario = "base"  # "base", "missing_data", "trajectory_noise"

    # Missing-data probability (used only when scenario == "missing_data")
    # Recommended discrete set (per your experiments): 0.20, 0.40, 0.60, 0.80
    # Must be 0.0 for other scenarios (enforced by validate_config).
    missing_data_p = 0.00

    # Trajectory-noise probability/ratio (used only when scenario == "trajectory_noise")
    # Recommended discrete set: 0.02, 0.04, 0.06, 0.08, 0.10, 0.20, 0.30, 0.40
    # Must be 0.0 for other scenarios (enforced by validate_config).
    trajectory_noise_p = 0.00

    # Number of training/testing episodes (each episode typically simulates task arrivals and offloading).
    total_episodes = 500

    # ======================================================================
    # 3) Infrastructure: RSUs and servers
    # ======================================================================

    # Number of total edge servers in the system 
    NUM_EDGE_SERVERS = 0

    # Number of edge servers that are connected to each RSU.
    RSUs_EDGE_SERVERS = (6, 7)

    # RSU coverage radius range in meters 
    RSU_radius = (850, 1000)

    # Number of cloud servers (remote compute resources).
    NUM_CLOUD_SERVERS = 2

    # Total number of servers available (edge + cloud).
    serverNo = NUM_EDGE_SERVERS + NUM_CLOUD_SERVERS

    # ======================================================================
    # 4) Workload: vehicles and tasks
    # ======================================================================

    # Number of vehicles in the episode.
    num_vehicles = 3

    # Number of tasks generated per vehicle per episode.
    Vehicle_taskno = 600

    # Total tasks per episode (derived).
    taskno = Vehicle_taskno * num_vehicles

    # Task arrival rate per vehicle (tasks/second), used for stochastic generation.
    TASK_ARRIVAL_RATE_range = (0.8, 1.0)  # task/s

    # Input data size per task (Mb).
    TASK_SIZE_RANGE = (100, 1000)  # Mb

    # Output/result size per task (Mb).
    task_result_size = 10  # Mb

    # Task compute demand (MIPS) bounds.
    # You later use these bounds to parameterize a normal distribution for failure-rate mapping.
    Low_demand, High_demand = 1, 100  # MIPS (Normal(mean=50, std=16) implied)

    # ======================================================================
    # 5) Network model
    # ======================================================================
    # Propagation speed for RSU-to-RSU links (used in propagation delay term)
    network_speed = 2e8  # m/s

    # RSU-to-RSU link bandwidth range (used in transmission delay term)
    RSU_LINK_BANDWIDTH_RANGE = (200, 500)  # Mb/s

    # Baseline RSU-to-RSU link failure probability range (used for retransmission delay)
    link_failure_rate_range = (0.1, 0.5)

    # Queueing delay scaling factor based on destination RSU load
    Q_alpha = 5

    # Load-dependent amplification factor for RSU-to-RSU link failure probability
    beta = 0.2

    # Cache timeout for task results (seconds) if caching is used.
    task_timeout_caching = 100

    # RSU-to-cloud backhaul bandwidth (Mb/s).
    rsu_to_cloud_bandwidth = 80  # Mb/s

    # ======================================================================
    # 6) RL hyperparameters — DQN (Global/Level-1 and Local/Level-2)
    # ======================================================================

    # ------------------------------
    # Level-1 (Global) DQN settings
    # ------------------------------
    global_hidden_layers_dqn = [128, 64]   # Q-network MLP hidden units
    global_af_dqn = "relu"                # Activation function
    global_lr_dqn = 3e-4                  # Learning rate
    global_gamma_dqn = 0.85               # Discount factor
    global_tau_dqn = 0.005                # Soft update factor for target network
    global_buffer_capacity_dqn = 500_000  # Replay buffer capacity
    global_batch_size_dqn = 256           # Minibatch size
    global_epsilon_start_dqn = 1.0        # Initial exploration probability
    global_epsilon_end_dqn = 0.01         # Final exploration probability
    global_epsilon_decay_dqn = 400        # Decay schedule (steps/episodes depending on your code)

    # ------------------------------
    # Level-2 (Local) DQN settings
    # ------------------------------
    local_hidden_layers_dqn = [128, 64]
    local_af_dqn = "relu"
    local_lr_dqn = 5e-4
    local_gamma_dqn = 0.90
    local_tau_dqn = 0.005
    local_buffer_capacity_dqn = 200_000
    local_batch_size_dqn = 256
    local_epsilon_start_dqn = 1.0
    local_epsilon_end_dqn = 0.01
    local_epsilon_decay_dqn = 300

    # ======================================================================
    # 7) RL hyperparameters — PPO (Global/Level-1 and Local/Level-2)
    # ======================================================================

    # ------------------------------
    # Level-1 (Global) PPO settings
    # ------------------------------
    global_hidden_layers_ppo = [256, 128]
    global_af_ppo = "tanh"
    global_actor_lr_ppo = 3e-4
    global_critic_lr_ppo = 3e-4
    global_gamma_ppo = 0.90
    global_clip_eps_ppo = 0.2             # PPO clipping epsilon
    global_k_epochs_ppo = 3               # PPO epochs per update
    global_batch_size_ppo = 128
    global_entropy_coef_ppo = 0.002       # Exploration encouragement via entropy bonus
    global_reward_scale_ppo = 1           # Optional scaling for reward magnitude
    global_gae_lambda_ppo = 0.95          # GAE lambda
    global_value_loss_coef_ppo = 0.5      # Value loss coefficient
    global_max_grad_norm_ppo = 0.5        # Gradient clipping

    # ------------------------------
    # Level-2 (Local) PPO settings
    # ------------------------------
    local_hidden_layers_ppo = [64, 32]
    local_af_ppo = "tanh"
    local_actor_lr_ppo = 1e-4
    local_critic_lr_ppo = 5e-4
    local_gamma_ppo = 0.90
    local_clip_eps_ppo = 0.2
    local_k_epochs_ppo = 2
    local_batch_size_ppo = 64
    local_entropy_coef_ppo = 0.01
    local_reward_scale_ppo = 1
    local_gae_lambda_ppo = 0.95
    local_value_loss_coef_ppo = 0.5
    local_max_grad_norm_ppo = 0.5

    # ======================================================================
    # 8) Reliability model parameters (Edge vs Cloud)
    # ======================================================================

    # Edge reliability: base failure probability for each state (low/med/high).
    # These are interpreted as failure-probability percentiles used later to derive failure rates.
    INITIAL_FAILURE_PROB_LOW_EDGE = 0.0001
    INITIAL_FAILURE_PROB_HIGH_EDGE = 0.79
    INITIAL_FAILURE_PROB_MED_EDGE = 0.55

    # Interval width around the base failure probability for:
    # - homogeneous   : smaller spread across nodes
    # - heterogeneous : larger spread across nodes
    HOMOGENEOUS_INTERVAL_EDGE = 0.10
    HETEROGENEOUS_INTERVAL_EDGE = 0.20

    # Edge processing capacity range (MIPS).
    EDGE_PROCESSING_FREQ_RANGE = (10, 15)  # MIPS

    # Cloud reliability: base failure probabilities (typically much lower than edge).
    INITIAL_FAILURE_PROB_LOW_CLOUD = 1e-6
    INITIAL_FAILURE_PROB_HIGH_CLOUD = 7.9e-6
    INITIAL_FAILURE_PROB_MED_CLOUD = 5.5e-6

    # Interval widths for cloud (small values due to very low failure probabilities).
    HOMOGENEOUS_INTERVAL_CLOUD = 1e-6
    HETEROGENEOUS_INTERVAL_CLOUD = 2e-6

    # Cloud processing capacity range (MIPS).
    CLOUD_PROCESSING_FREQ_RANGE = (30, 60)  # MIPS

    # Mapping from state IDs to labels used in FAILURE_STATE.
    STATES = {
        "S1": "low",
        "S2": "high",
        "S3": "med",
    }

    # ======================================================================
    # 9) Reliability helper methods
    # ======================================================================

    @staticmethod
    def compute_failure_probabilities():
        """
        Build failure-probability intervals for edge and cloud under:
        - homogeneous scenario type
        - heterogeneous scenario type

        Output format:
            {
              'edge':  {'homogeneous': {'low': (p0, p1), ...}, 'heterogeneous': {...}},
              'cloud': {'homogeneous': {...},               'heterogeneous': {...}}
            }
        """
        return {
            "edge": {
                "homogeneous": {
                    "low": (
                        parameters.INITIAL_FAILURE_PROB_LOW_EDGE,
                        parameters.INITIAL_FAILURE_PROB_LOW_EDGE + parameters.HOMOGENEOUS_INTERVAL_EDGE,
                    ),
                    "high": (
                        parameters.INITIAL_FAILURE_PROB_HIGH_EDGE,
                        parameters.INITIAL_FAILURE_PROB_HIGH_EDGE + parameters.HOMOGENEOUS_INTERVAL_EDGE,
                    ),
                    "med": (
                        parameters.INITIAL_FAILURE_PROB_MED_EDGE,
                        parameters.INITIAL_FAILURE_PROB_MED_EDGE + parameters.HOMOGENEOUS_INTERVAL_EDGE,
                    ),
                },
                "heterogeneous": {
                    "low": (
                        parameters.INITIAL_FAILURE_PROB_LOW_EDGE,
                        parameters.INITIAL_FAILURE_PROB_LOW_EDGE + parameters.HETEROGENEOUS_INTERVAL_EDGE,
                    ),
                    "high": (
                        parameters.INITIAL_FAILURE_PROB_HIGH_EDGE,
                        parameters.INITIAL_FAILURE_PROB_HIGH_EDGE + parameters.HETEROGENEOUS_INTERVAL_EDGE,
                    ),
                    "med": (
                        parameters.INITIAL_FAILURE_PROB_MED_EDGE,
                        parameters.INITIAL_FAILURE_PROB_MED_EDGE + parameters.HETEROGENEOUS_INTERVAL_EDGE,
                    ),
                },
            },
            "cloud": {
                "homogeneous": {
                    "low": (
                        parameters.INITIAL_FAILURE_PROB_LOW_CLOUD,
                        parameters.INITIAL_FAILURE_PROB_LOW_CLOUD + parameters.HOMOGENEOUS_INTERVAL_CLOUD,
                    ),
                    "high": (
                        parameters.INITIAL_FAILURE_PROB_HIGH_CLOUD,
                        parameters.INITIAL_FAILURE_PROB_HIGH_CLOUD + parameters.HOMOGENEOUS_INTERVAL_CLOUD,
                    ),
                    "med": (
                        parameters.INITIAL_FAILURE_PROB_MED_CLOUD,
                        parameters.INITIAL_FAILURE_PROB_MED_CLOUD + parameters.HOMOGENEOUS_INTERVAL_CLOUD,
                    ),
                },
                "heterogeneous": {
                    "low": (
                        parameters.INITIAL_FAILURE_PROB_LOW_CLOUD,
                        parameters.INITIAL_FAILURE_PROB_LOW_CLOUD + parameters.HETEROGENEOUS_INTERVAL_CLOUD,
                    ),
                    "high": (
                        parameters.INITIAL_FAILURE_PROB_HIGH_CLOUD,
                        parameters.INITIAL_FAILURE_PROB_HIGH_CLOUD + parameters.HETEROGENEOUS_INTERVAL_CLOUD,
                    ),
                    "med": (
                        parameters.INITIAL_FAILURE_PROB_MED_CLOUD,
                        parameters.INITIAL_FAILURE_PROB_MED_CLOUD + parameters.HETEROGENEOUS_INTERVAL_CLOUD,
                    ),
                },
            },
        }

    @staticmethod
    def compute_failure_rates():
        """
        Convert failure-probability intervals into failure-rate intervals.

        Approach:
        - Use a Normal distribution over task demand (Low_demand..High_demand) to map probability intervals
          to percentile values via `norm.ppf`.
        - Convert percentile values to rates via inverse (1 / percentile_value).

        Output format mirrors compute_failure_probabilities().
        """
        failure_probs = parameters.compute_failure_probabilities()

        # Normal distribution parameters derived from demand bounds:
        # mean at the midpoint, std so that +/-3 std spans [Low_demand, High_demand]
        mean = (parameters.Low_demand + parameters.High_demand) / 2
        std = (parameters.High_demand - parameters.Low_demand) / 6

        def get_failure_rate_interval(prob_interval):
            # Translate probability interval into percentile values of the demand distribution.
            lower_percentile_value = norm.ppf(1 - prob_interval[0], loc=mean, scale=std)
            upper_percentile_value = norm.ppf(1 - prob_interval[1], loc=mean, scale=std)
            # Convert percentile values into rate bounds (project-specific interpretation).
            return (1 / lower_percentile_value, 1 / upper_percentile_value)

        def compute_all(rtype):
            return {
                "homogeneous": {
                    k: get_failure_rate_interval(v)
                    for k, v in failure_probs[rtype]["homogeneous"].items()
                },
                "heterogeneous": {
                    k: get_failure_rate_interval(v)
                    for k, v in failure_probs[rtype]["heterogeneous"].items()
                },
            }

        return {"edge": compute_all("edge"), "cloud": compute_all("cloud")}

    @staticmethod
    def compute_Alpha():
        """
        Compute alpha parameters used by your simulator (typically for time-varying failure or hazard models).

        For each failure-rate interval (r0, r1), alpha is computed as:
            alpha_0 = (r1 - r0) / taskno
            alpha_1 = r1

        Output format mirrors compute_failure_rates().
        """
        failure_rates = parameters.compute_failure_rates()

        def calc_alpha(rate_interval):
            return ((rate_interval[1] - rate_interval[0]) / parameters.taskno, rate_interval[1])

        def compute_all(rtype):
            return {
                "homogeneous": {k: calc_alpha(v) for k, v in failure_rates[rtype]["homogeneous"].items()},
                "heterogeneous": {k: calc_alpha(v) for k, v in failure_rates[rtype]["heterogeneous"].items()},
            }

        return {"edge": compute_all("edge"), "cloud": compute_all("cloud")}

    # ======================================================================
    # 10) Configuration validation (prevents mixed scenarios)
    # ======================================================================

    @staticmethod
    def validate_config(strict: bool = True):
        """
        Ensures scenario-related parameters are mutually exclusive.

        Rules:
        - base:
            missing_data_p == 0.0 and trajectory_noise_p == 0.0
        - missing_data:
            missing_data_p > 0.0 and trajectory_noise_p == 0.0
        - trajectory_noise:
            trajectory_noise_p > 0.0 and missing_data_p == 0.0

        Usage:
            from configuration import parameters as params
            params.validate_config(strict=True)
        """
        s = parameters.scenario
        md = float(parameters.missing_data_p)
        tn = float(parameters.trajectory_noise_p)

        def fail(msg: str):
            if strict:
                raise ValueError(f"[CONFIG ERROR] {msg}")
            print(f"[CONFIG WARNING] {msg}")

        if s == "base":
            if md != 0.0 or tn != 0.0:
                fail(
                    f'scenario="base" requires missing_data_p=0.0 and trajectory_noise_p=0.0 (got {md}, {tn}).'
                )
        elif s == "missing_data":
            if md <= 0.0:
                fail(f'scenario="missing_data" requires missing_data_p > 0.0 (got {md}).')
            if tn != 0.0:
                fail(f'scenario="missing_data" requires trajectory_noise_p=0.0 (got {tn}).')
        elif s == "trajectory_noise":
            if tn <= 0.0:
                fail(f'scenario="trajectory_noise" requires trajectory_noise_p > 0.0 (got {tn}).')
            if md != 0.0:
                fail(f'scenario="trajectory_noise" requires missing_data_p=0.0 (got {md}).')
        else:
            fail(f'Unknown scenario "{s}". Allowed: base, missing_data, trajectory_noise')
