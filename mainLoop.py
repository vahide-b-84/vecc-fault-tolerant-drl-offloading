# mainLoop.py

from Sumo_Graph import GraphNetwork
from task import Task
from EnvState import EnvironmentState
from params import params
import simpy
import pandas as pd
import os
from RSU_Vehicle_Setup import RSU_and_Vehicle_setup
from Global_model import global_model
import traci
from save import save_params_and_logs  # Save parameters and logs to files


class MainLoop:
    """
    Main simulation loop that orchestrates:
    - SUMO traffic simulation
    - SimPy discrete-event simulation
    - Task generation and submission
    - Global (Level-1) and Local (Level-2) decision models
    """

    def __init__(self):

        # Load graph/network data (previously extracted from SUMO and saved as JSON)
        self.network = GraphNetwork()
        self.network.load_graph()
        # self.network.plot_graph()  # Optional visualization

        # Initialize SimPy environment
        self.env = simpy.Environment()

        # Event used to terminate one episode iteration
        self.iteration_complete_event = self.env.event()

        # Setup RSUs and vehicles based on the loaded graph
        self.RSU_and_Vehicle = RSU_and_Vehicle_setup(self.env, self.network)

        # Initialize environment state (servers, RSUs, tasks, metrics)
        self.env_state = EnvironmentState(self.env, self.RSU_and_Vehicle)

        # Initialize the global (Level-1) decision model
        self.G_model = global_model(self.env, self.env_state)

        # Load task parameters
        self.task_params_df = pd.read_excel('task_parameters.xlsx')

        # Minimum computation demand used for scheduling delays
        params.min_computation_demand = self.task_params_df['Computation_Demand'].min()

        # Task inter-arrival times
        self.inter_arrival_times = self.task_params_df['Interarrival_Time'].tolist()

        # Episode-level bookkeeping
        self.this_episode = 0
        self.total_episodes = params.total_episodes
        self.taskCounter = 1

        # RSU task distribution log (used in flat mode)
        self.rsu_taskcount_log = []  # (episode, RSU_0, RSU_1, ...)

    def EP(self):
        """
        Run the full experiment over multiple episodes.
        """

        self.this_episode = 0

        while self.this_episode < self.total_episodes:

            self.this_episode += 1
            print(f"Starting episode {self.this_episode}...")

            # Reset environment, models, and states
            self.reset_setting()

            # Start task generation / processing
            self.env.process(self.iteration())

            # Start SUMO simulation as a parallel SimPy process
            self.env.process(self.RSU_and_Vehicle.Start_SUMO(use_gui=False))

            # Run until the iteration signals completion
            self.env.run(until=self.iteration_complete_event)

            # Close SUMO connection after each episode
            traci.close()

        # Save all logs after finishing all episodes
        self.save_Logs()

    def iteration(self):
        """
        One episode task-generation loop.
        """

        # Generate tasks according to inter-arrival times
        for inter_arrival_time in self.inter_arrival_times:

            yield self.env.timeout(inter_arrival_time)

            # Create a new task
            task = Task(
                self.env,
                self.env_state,
                self.taskCounter,
                self.RSU_and_Vehicle.vehicles,
                self.task_params_df
            )

            # Register task in environment state
            self.env_state.add_task(task)

            # Submit task asynchronously
            self.env.process(self.task_submition(task))

            self.taskCounter += 1

        # Wait until all tasks have been forwarded to their selected RSUs
        while True:
            all_tasks_ready = all(
                task.selected_rsu_start_time is not None
                for task in self.env_state.tasks.values()
            )
            if all_tasks_ready:
                break
            else:
                yield self.env.timeout(5)

        # ---------- Process remaining pending tasks ----------
        if params.Flat_mode:
            # Flat architecture: only global model
            yield self.env.process(
                self.G_model.process_pendingList_and_log_result(self.this_episode)
            )
            self.log_rsu_task_distribution(self.this_episode)

        else:
            # Two-level architecture: RSUs + global model
            processes = []

            for rsu in self.RSU_and_Vehicle.RSUs.values():
                processes.append(
                    self.env.process(
                        rsu.process_pendingList_and_log_result(self.this_episode)
                    )
                )

            if params.model_summary in ["dqn", "ppo"]:
                processes.append(
                    self.env.process(
                        self.G_model.process_pendingList_and_log_result(self.this_episode)
                    )
                )
            elif params.model_summary in ["NoForwarding", "greedy"]:
                processes.append(
                    self.env.process(
                        self.G_model.simple_process_pendingList_and_log_result(self.this_episode)
                    )
                )
            else:
                raise ValueError(f"Unknown model summary: {params.model_summary}")

            yield self.env.all_of(processes)

        print("the task iteration completed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # Signal episode completion
        self.iteration_complete_event.succeed()

    def task_submition(self, task):
        """
        Handle task submission from vehicle to RSU and start execution.
        """

        # Wait until the vehicle is within coverage of an RSU
        max_wait_time = 1000
        waited_time = 0

        while task.vehicle.Current_RSU is None and waited_time < max_wait_time:
            yield self.env.timeout(1)
            waited_time += 1

        if task.vehicle.Current_RSU is None:
            print(f"Task {task.id} could not find a valid RSU.")
            return

        # Register submission info
        task.original_RSU = task.vehicle.Current_RSU
        task.submitted_time = self.env.now
        task.vehicle.add_pending_task(task)

        if params.Flat_mode:
            # Flat global decision: RSU + execution plan together
            if params.model_summary in ["dqn", "ppo"]:
                selected_RSU, X, Y, Z = self.G_model.Recommend_action(
                    task, self.this_episode
                )
            else:
                raise ValueError(f"Unknown model summary: {params.model_summary}")

            yield self.env.process(selected_RSU.receive_task(task))

        else:
            # Two-level decision
            if params.model_summary in ["dqn", "ppo"]:
                selected_RSU = self.G_model.Recommend_RSU(task, self.this_episode)
            elif params.model_summary == "NoForwarding":
                selected_RSU = self.G_model.simple_Recommend_RSU(task)
            elif params.model_summary == "greedy":
                selected_RSU = self.G_model.greedy_Recommend_RSU(task)
            else:
                raise ValueError(f"Unknown model summary: {params.model_summary}")

            yield self.env.process(selected_RSU.receive_task(task))

            # Local RSU selects execution servers
            X, Y, Z = selected_RSU.Recommend_XYZ(task, self.this_episode)

        # Start task execution
        self.env.process(task.execute_task(X, Y, Z))

    def clear_logs(self, rsu_logs_dict, rsu_assignments_dict):
        """
        Clear in-memory logs after saving them to disk.
        """
        self.G_model.log_data.clear()
        self.G_model.task_Assignments_info.clear()

        for rsu_id in rsu_logs_dict:
            rsu_logs_dict[rsu_id].clear()
            rsu_assignments_dict[rsu_id].clear()

    def reset_setting(self):
        """
        Reset environment and models at the beginning of each episode.
        """
        self.taskCounter = 1

        del self.env
        self.env = simpy.Environment()

        self.env_state.reset(self.env, self.this_episode)
        self.G_model.reset(self.env, self.this_episode)

        self.iteration_complete_event = self.env.event()

    def save_Logs(self):
        """
        Save global and RSU-level logs after all episodes finish.
        """
        if not params.Flat_mode:
            rsu_logs_dict, rsu_assignments_dict = self.RSU_and_Vehicle.extract_rsu_logs_and_assignments()

            save_params_and_logs(
                params,
                self.G_model.log_data,
                self.G_model.task_Assignments_info,
                rsu_logs_dict,
                rsu_assignments_dict
            )

            self.clear_logs(rsu_logs_dict, rsu_assignments_dict)

        else:
            results_dir = save_params_and_logs(
                params,
                self.G_model.log_data,
                self.G_model.task_Assignments_info,
                None,
                None
            )
            self.save_RSU_logs(results_dir)

    # -------------------- Flat-mode RSU statistics --------------------

    def log_rsu_task_distribution(self, episode: int):
        """
        Log number of tasks handled by each RSU in flat mode.
        """
        rsus = self.RSU_and_Vehicle.RSUs
        rsu_ids = sorted(rsus.keys())

        row = [episode]
        for rid in rsu_ids:
            row.append(rsus[rid].taskCounter)

        self.rsu_taskcount_log.append(tuple(row))

    def save_RSU_logs(self, output_dir: str):
        """
        Save RSU task distribution per episode to CSV (flat mode).
        """
        rsu_ids = sorted(self.RSU_and_Vehicle.RSUs.keys())
        cols = ["episode"] + [f"{rid}" for rid in rsu_ids]

        df = pd.DataFrame(self.rsu_taskcount_log, columns=cols)

        out_path = os.path.join(output_dir, "rsu_taskcount_per_episode.csv")
        df.to_csv(out_path, index=False)
