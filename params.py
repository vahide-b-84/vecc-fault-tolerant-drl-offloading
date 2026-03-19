# params.py
# Centralized project parameters wrapper.
# This module exposes a single class (`params`) that collects configuration values
# from `configuration.parameters`, plus a few derived values extracted from SUMO files.

from configuration import parameters
parameters.validate_config(strict=True)  # Validate configuration at import-time (fail fast if invalid)

from env_extractors import extract_max_speed_from_rou, estimate_max_e2e_delay  # Derived environment bounds (SUMO-based)

class params:
    # ----------------------------
    # High-level experiment setup
    # ----------------------------
    model_summary = parameters.model_summary  # Options: "dqn", "ppo", "greedy", "NoForwarding"
    Flat_mode = parameters.Flat_mode  # True/False (flat unified controller vs. 2-level architecture)
    missing_data_p = parameters.missing_data_p  # Probability used in the "missing_data" scenario
    trajectory_noise_p = parameters.trajectory_noise_p  # Probability used in the "trajectory_noise" scenario
    scenario = parameters.scenario  # Scenario tag string (e.g., "base", "trajectory_noise", "missing_data")

    # Set later (e.g., by MainLoop after reading task_parameters.xlsx)
    min_computation_demand = None  # Minimum computation_demand across tasks (used as a small SimPy polling step)

    # Scenario metadata
    SCENARIO_TYPE = parameters.SCENARIO_TYPE  # e.g., "heterogeneous" / "homogeneous"
    FAILURE_STATE = parameters.FAILURE_STATE  # e.g., "low" / "med" / "high"

    # Failure/load model parameters
    Alpha = parameters.compute_Alpha()  # Precomputed Alpha tables (edge/cloud, scenario/state)
    alpha_edge = (None, None)  # Will be set at runtime based on SCENARIO_TYPE and FAILURE_STATE
    alpha_cloud = (None, None)  # Will be set at runtime based on SCENARIO_TYPE and FAILURE_STATE

    # ----------------------------
    # Infrastructure sizing
    # ----------------------------
    NUM_EDGE_SERVERS = parameters.NUM_EDGE_SERVERS
    NUM_CLOUD_SERVERS = parameters.NUM_CLOUD_SERVERS
    RSUs_EDGE_SERVERS = parameters.RSUs_EDGE_SERVERS  # Edge server count range per RSU (used in graph generation)
    RSU_radius = parameters.RSU_radius  # RSU coverage radius range (used in graph generation)

    # ----------------------------
    # Workload sizing
    # ----------------------------
    num_vehicles = parameters.num_vehicles
    task_result_size = parameters.task_result_size
    TASK_ARRIVAL_RATE_range = parameters.TASK_ARRIVAL_RATE_range  # Per-vehicle task arrival rate range
    taskno = parameters.taskno
    Vehicle_taskno = parameters.Vehicle_taskno  # Number of tasks per vehicle (used in GraphNetwork task-time generation)
    total_episodes = parameters.total_episodes

    # Task characteristics
    TASK_SIZE_RANGE = parameters.TASK_SIZE_RANGE
    Low_demand, High_demand = parameters.Low_demand, parameters.High_demand

    # Server capabilities
    EDGE_PROCESSING_FREQ_RANGE = parameters.EDGE_PROCESSING_FREQ_RANGE
    CLOUD_PROCESSING_FREQ_RANGE = parameters.CLOUD_PROCESSING_FREQ_RANGE

    # Derived global bounds from SUMO route/network files (used for normalization / scaling elsewhere)
    MAX_VEHICLE_SPEED = extract_max_speed_from_rou()
    Max_e2e_Delay = estimate_max_e2e_delay()

    # ----------------------------
    # Network parameters
    # ----------------------------
    rsu_to_cloud_bandwidth = parameters.rsu_to_cloud_bandwidth  # Bandwidth used for RSU->cloud transfers
    network_speed = parameters.network_speed  # Generic network speed parameter (if used in delay estimation)
    RSU_LINK_BANDWIDTH_RANGE = parameters.RSU_LINK_BANDWIDTH_RANGE  # Range for RSU-to-RSU link bandwidths (graph generation)
    task_timeout_caching = parameters.task_timeout_caching  # TTL for cached results at RSUs (if enabled)
    link_failure_rate_range = parameters.link_failure_rate_range  # Range for RSU link failure rates (graph generation)

    # Queueing / delay model parameters (used in EnvironmentState or delay formulas)
    Q_alpha = parameters.Q_alpha
    beta = parameters.beta

    # ----------------------------
    # RL hyperparameters: Global DQN (Level-1)
    # ----------------------------
    Global_DQN = {
        'activation': parameters.global_af_dqn,
        'hidden_layers': parameters.global_hidden_layers_dqn,
        'lr': parameters.global_lr_dqn,
        'gamma': parameters.global_gamma_dqn,
        'tau': parameters.global_tau_dqn,
        'buffer_capacity': parameters.global_buffer_capacity_dqn,
        'batch_size': parameters.global_batch_size_dqn,
        'epsilon_start': parameters.global_epsilon_start_dqn,
        'epsilon_end': parameters.global_epsilon_end_dqn,
        'epsilon_decay': parameters.global_epsilon_decay_dqn,
    }

    # ----------------------------
    # RL hyperparameters: Local DQN (Level-2)
    # ----------------------------
    Local_DQN = {
        'activation': parameters.local_af_dqn,
        'hidden_layers': parameters.local_hidden_layers_dqn,
        'lr': parameters.local_lr_dqn,
        'gamma': parameters.local_gamma_dqn,
        'tau': parameters.local_tau_dqn,
        'buffer_capacity': parameters.local_buffer_capacity_dqn,
        'batch_size': parameters.local_batch_size_dqn,
        'epsilon_start': parameters.local_epsilon_start_dqn,
        'epsilon_end': parameters.local_epsilon_end_dqn,
        'epsilon_decay': parameters.local_epsilon_decay_dqn,
    }

    # ----------------------------
    # RL hyperparameters: Global PPO (Level-1)
    # ----------------------------
    Global_PPO = {
        'activation': parameters.global_af_ppo,
        'hidden_layers': parameters.global_hidden_layers_ppo,
        'actor_lr': parameters.global_actor_lr_ppo,
        'critic_lr': parameters.global_critic_lr_ppo,
        'gamma': parameters.global_gamma_ppo,
        'clip_eps': parameters.global_clip_eps_ppo,
        'k_epochs': parameters.global_k_epochs_ppo,
        'batch_size': parameters.global_batch_size_ppo,
        'entropy_coef': parameters.global_entropy_coef_ppo,

        # Kept only for interface compatibility (not used by PPO action selection)
        'epsilon_start': 1,      # not used in PPO (compatibility only)
        'epsilon_end': 0.01,     # not used in PPO (compatibility only)
        'epsilon_decay': 400,    # not used in PPO (compatibility only)

        # PPO extras (scaling/stability)
        'reward_scale': parameters.global_reward_scale_ppo,
        'gae_lambda': parameters.global_gae_lambda_ppo,
        'value_loss_coef': parameters.global_value_loss_coef_ppo,
        'max_grad_norm': parameters.global_max_grad_norm_ppo,
    }

    # ----------------------------
    # RL hyperparameters: Local PPO (Level-2)
    # ----------------------------
    Local_PPO = {
        'activation': parameters.local_af_ppo,
        'hidden_layers': parameters.local_hidden_layers_ppo,
        'actor_lr': parameters.local_actor_lr_ppo,
        'critic_lr': parameters.local_critic_lr_ppo,
        'gamma': parameters.local_gamma_ppo,
        'clip_eps': parameters.local_clip_eps_ppo,
        'k_epochs': parameters.local_k_epochs_ppo,
        'batch_size': parameters.local_batch_size_ppo,
        'entropy_coef': parameters.local_entropy_coef_ppo,

        # Kept only for interface compatibility (not used by PPO action selection)
        'epsilon_start': 1,      # not used in PPO (compatibility only)
        'epsilon_end': 0.01,     # not used in PPO (compatibility only)
        'epsilon_decay': 400,    # not used in PPO (compatibility only)

        # PPO extras (scaling/stability)
        'reward_scale': parameters.local_reward_scale_ppo,
        'gae_lambda': parameters.local_gae_lambda_ppo,
        'value_loss_coef': parameters.local_value_loss_coef_ppo,
        'max_grad_norm': parameters.local_max_grad_norm_ppo,
    }

    # ----------------------------
    # Active config selection based on chosen algorithm
    # ----------------------------
    if model_summary == "ppo":
        # If global-level algorithm is PPO, both Global/Local use PPO parameter dicts
        Global = Global_PPO
        Local = Local_PPO
    else:  # If level-1 model is selected as: dqn / greedy / NoForwarding
        # Default local/global param dicts are DQN-style
        Global = Global_DQN
        Local = Local_DQN
