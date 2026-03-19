# Global_model.py: Level-1 model
import torch
import numpy as np
import math
from params import params
import random
from DQN_template import DQNAgent
from PPO_template import PPOAgent
import os
import csv

class global_model:
    def __init__(self, env, env_state):
        self.env = env
        self.env_state = env_state

        # Global state/action dimensions depend on the number of RSUs in the environment
        num_rsus = len(self.env_state.RSU_and_Vehicle.RSUs)
        self.num_states = 5 * num_rsus + num_rsus * (num_rsus - 1) + 3
        self.num_actions = num_rsus

        # Flat-mode merges Level-1 (RSU selection) and Level-2 (server+z selection) into one action
        if(params.Flat_mode):
            self.Set_Flat_setting()

        # Action selection behavior (e.g., DQN select_action may support softmax sampling)
        self.use_softmax=True

        # Epsilon schedule for exploration (used by DQN / and by select_action wrapper)
        self.epsilon_start = params.Global['epsilon_start']
        self.epsilon_end = params.Global['epsilon_end']
        self.epsilon_decay = params.Global['epsilon_decay']
        self.current_epsilon=self.epsilon_start

        # Current global state/action
        self.G_state = []
        self.G_action = []

        # tempbuffer[taskCounter] = (s, a, r, s_next)
        # reward and next-state are filled later when the task outcome is known
        self.tempbuffer = {}
        self.taskCounter = 1
        self.pendingList = []  # list of (task_id, taskCounter) awaiting reward resolution

        # Episode-level metrics and logs
        self.rewardsAll = []
        self.ep_reward_list = []
        self.ep_delay_list = []
        self.avg_reward_list = []
        self.episodic_reward = 0
        self.episodic_delay = 0
        self.log_data = []
        self.task_Assignments_info = []

        self.total_episodes = params.total_episodes

        # Per-run reproducible RNG used in missing_data scenario (avoid recreating RNG per task)
        self.net_rng = random.Random(self.total_episodes)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if params.model_summary == "ppo":
            # Global PPO agent (on-policy)
            self.agent = PPOAgent(
                num_states=self.num_states,
                num_actions=self.num_actions,
                hidden_layers=params.Global['hidden_layers'],
                device=device,
                gamma=params.Global['gamma'],
                actor_lr=params.Global['actor_lr'],
                critic_lr=params.Global['critic_lr'],
                clip_eps=params.Global['clip_eps'],
                k_epochs=params.Global['k_epochs'],
                batch_size=params.Global['batch_size'],
                entropy_coef=params.Global['entropy_coef'],
                reward_scale=params.Global.get('reward_scale', 1.0),
                gae_lambda=params.Global.get('gae_lambda', 0.95),
                value_loss_coef=params.Global.get('value_loss_coef', 0.5),
                max_grad_norm=params.Global.get('max_grad_norm', 0.5),
                activation=params.Global['activation'],
            )
            print("Global model: using PPOAgent")

        elif params.model_summary == "dqn":
            # Global DQN agent (off-policy)
            self.agent = DQNAgent(
                num_states=self.num_states,
                num_actions=self.num_actions,
                device=device,
                gamma=params.Global['gamma'],
                lr=params.Global['lr'],
                tau=params.Global['tau'],
                buffer_size=params.Global['buffer_capacity'],
                batch_size=params.Global['batch_size'],
                activation=params.Global['activation'],
                hidden_layers=params.Global['hidden_layers'],
            )
            print("Global model: using DQNAgent")

    def update_episode_epsilon(self,episode):
        # Linear epsilon decay + optional "bump" if recent performance drops
        epsilon = max(self.epsilon_end, self.epsilon_start - (episode / self.epsilon_decay))
        if len(self.ep_reward_list) >= 40:
            recent_rewards = self.ep_reward_list[-40:]
            avg_recent = np.mean(recent_rewards)
            last_reward = self.ep_reward_list[-1]
            drop_ratio = max(0.0, (avg_recent - last_reward) / avg_recent)
            if drop_ratio > 0.4:
                epsilon = max(epsilon, 0.25)
            elif drop_ratio > 0.2:
                epsilon = max(epsilon, 0.20)
            elif drop_ratio > 0.05:
                epsilon = max(epsilon, 0.15)

        self.current_epsilon = epsilon

    def add_train(self, this_episode):
        # Resolve rewards for pending tasks; store transitions and (for DQN) train online
        removeList = []
        if params.model_summary == "ppo":
            # PPO: store transitions; policy update happens at end of episode
            for taskid, task_counter in self.pendingList:
                reward, delay = self.calcReward(taskid)
                if reward is not None:
                    # Update episode aggregates
                    self.episodic_reward += reward
                    self.episodic_delay += delay
                    self.rewardsAll.append(reward)

                    # Fill reward into stored transition
                    temp = list(self.tempbuffer[task_counter])
                    temp[2] = reward
                    self.tempbuffer[task_counter] = tuple(temp)
                    s, a, r, s_ = self.tempbuffer[task_counter]

                    # missing_data scenario: drop some transitions before they reach the learner
                    if params.scenario=="missing_data":
                        if self.net_rng.random() > params.missing_data_p:
                            self.sent_cnt += 1
                            self.agent.store_transition(s, a, r, s_ ,done=False)
                            # on-policy update step at the end of episode
                        else:
                            self.drop_cnt += 1
                    else:
                        self.agent.store_transition(s, a, r, s_ ,done=False)
                        # on-policy update step at the end of episode

                    removeList.append((taskid, task_counter))

        else: 
            # DQN: store transitions and train online
            for taskid, task_counter in self.pendingList:
                reward, delay = self.calcReward(taskid)
                if reward is not None:
                    self.episodic_reward += reward
                    self.episodic_delay += delay
                    self.rewardsAll.append(reward)

                    # Fill reward into stored transition
                    temp = list(self.tempbuffer[task_counter])
                    temp[2] = reward
                    self.tempbuffer[task_counter] = tuple(temp)
                    s, a, r, s_ = self.tempbuffer[task_counter]

                    # missing_data scenario: drop some transitions before they reach the learner
                    if params.scenario=="missing_data":
                        if self.net_rng.random() > params.missing_data_p:
                            self.sent_cnt += 1
                            self.agent.store_transition((s, a, r, s_))
                            self.agent.train_step()
                        else:
                            self.drop_cnt += 1
                    else:
                        self.agent.store_transition((s, a, r, s_))
                        self.agent.train_step()

                    removeList.append((taskid, task_counter))

        # Remove processed tasks and log assignment info
        for t in removeList:
            self.pendingList.remove(t)
            task = self.env_state.get_task_by_id(t[0])
            if params.Flat_mode:
                self.log_flat_data(task, this_episode)
            else:
                self.log_level1_data(task,this_episode)

    def calcReward(self, taskID):
        # Global reward is based on end-to-end delay and success/failure w.r.t. vehicle deadline
        task = self.env_state.get_task_by_id(taskID)
        reward=None
        delay=None
        if task is None:
            print("None task!")
            input("press Enter!")
            return None, None

        submitted_time = task.submitted_time
        deadline_flag = task.deadline_flag  # 'S' /'F'
        execution_flag = task.execution_status_flag  # 's' /'f'
        if submitted_time is None:
            print("task not submited yet in global model processing")
            input("press Enter!")
            return None, None

        # Delay definition depends on whether the deadline was met or missed
        if deadline_flag == 'S':
            delay = task.delivered_time - submitted_time
        elif deadline_flag == 'F':
            delay = task.timeout_time - submitted_time
        else: # deadline_flag == 'N': not reached deadline and not receive result yet
            return None, None  # not reached deadline and not receive result yet

        # Reward shaping parameters (Level-1)
        max_success_reward = 25.0
        min_success_reward = 5 # lower bound for success reward
        reward_decay_scale = 100.0  # key parameter controlling reward decay with delay
        failure_penalty_weight = 10.0
        min_failure_penalty = -3.0
        max_failure_penalty = -150.0

        # Successful task (completed before deadline)
        if execution_flag == 's' and deadline_flag == 'S':
            task.final_status_flag = "s"
            reward = min_success_reward + (max_success_reward - min_success_reward) * math.exp(-delay / reward_decay_scale)

        # Failed task (either execution failed or deadline missed)
        elif execution_flag == 'f' or deadline_flag == 'F':
            task.final_status_flag = "f"
            reward = -failure_penalty_weight * delay
            reward = max(min(reward, min_failure_penalty), max_failure_penalty)
        else:
            print("execution_flag:", execution_flag, "deadline_flag:", deadline_flag)
            input("press Enter!: None--------------------------------------------------")

        return reward, delay

    def Recommend_RSU(self, task, this_episode):
        # Choose an RSU (Level-1 decision) based on the current global state
        self.G_state = self.env_state.get_state(task)

        # Attach next-state to the previous transition and process any resolved rewards
        if self.taskCounter>1:
            tempx=list(self.tempbuffer[self.taskCounter-1])
            tempx[3]=self.G_state
            self.tempbuffer[self.taskCounter-1]=tuple(tempx)
            self.add_train(this_episode) 

        RSU_index = self.agent.select_action(self.G_state, self.current_epsilon, self.use_softmax, temperature=1.5)
        rsu_id = f"RSU_{RSU_index}"
        task.selected_RSU= self.env_state.RSU_and_Vehicle.RSUs[rsu_id]

        # Store transition; reward will be filled later
        self.tempbuffer[self.taskCounter] = (self.G_state, RSU_index, None, None)
        self.pendingList.append((task.id, self.taskCounter))
        self.taskCounter += 1
        
        return task.selected_RSU
         
    def process_pendingList_and_log_result(self, this_episode):
        # Finalize last transition's next-state, then drain pendingList until all rewards resolve
        print("Global model: process pending List! the last taskCounter:", self.taskCounter - 1)
        s, a, r, s_next = self.tempbuffer[self.taskCounter - 1]
        s_next = self.G_state
        self.tempbuffer[self.taskCounter - 1] = (s, a, r, s_next)

        while self.pendingList:
            yield self.env.timeout(params.min_computation_demand)
            self.add_train(this_episode)

        # PPO: update policy at end of episode
        if params.model_summary == "ppo":
            self.agent.train_step()

        # Episode-level logging (moving average uses last 40 episodes)
        self.ep_reward_list.append(self.episodic_reward)
        self.ep_delay_list.append(self.episodic_delay)
        avg_reward = np.mean(self.ep_reward_list[-40:])
        avg_delay = np.mean(self.ep_delay_list[-40:])
        self.log_data.append((this_episode, avg_reward, self.episodic_reward, avg_delay,self.episodic_delay))

        print("Global Model: Episode * {} * Avg Reward is ==> {}".format(this_episode, avg_reward), "This episode:", self.episodic_reward)
        self.avg_reward_list.append(avg_reward)

        if params.scenario == "missing_data":
            # ================= DROP RATIO LOGGING =================
            total = self.drop_cnt + self.sent_cnt
            drop_ratio = (self.drop_cnt / total) if total else 0.0

            # Results folder structure:
            # <SCENARIO_TYPE>_results/<scenario_folder>/<model_p>/
            base_folder = f"{params.SCENARIO_TYPE}_results"

            # Subfolder name example: dqn_0_30 (example)
            p = getattr(params, "missing_data_p", 0.0)
            scenario_folder = "missing_data"
            flat_suffix = "_flat" if getattr(params, "Flat_mode", False) else ""
            # Subfolder name example: dqn_0_30 (example)
            subfolder_name = f"{params.model_summary}{flat_suffix}_{p:.2f}".replace(".", "_")
            save_dir = os.path.join(base_folder, scenario_folder, subfolder_name)
            os.makedirs(save_dir, exist_ok=True)

            # CSV file path
            csv_path = os.path.join(
                save_dir,
                f"DROP_Ratio_{params.model_summary}.csv"
            )

            # Write header only once
            write_header = not os.path.exists(csv_path)
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "episode",
                        "sent_transitions",
                        "dropped_transitions",
                        "drop_ratio"
                    ])

                writer.writerow([
                    this_episode,
                    self.sent_cnt,
                    self.drop_cnt,
                    round(drop_ratio, 4)                
                ])

    def reset(self, new_SimPy_env,episode):
        # Reset per-episode buffers/counters
        self.env = new_SimPy_env
        self.episodic_reward = 0
        self.episodic_delay = 0
        self.tempbuffer = {}
        self.taskCounter = 1
        self.pendingList.clear()

        # Update epsilon for the new episode
        self.update_episode_epsilon(episode)

        # Missing-data counters
        self.drop_cnt = 0
        self.sent_cnt = 0
                
    #--------------------- methods for no existence of model level-1 mode "greedy"/"NoForwarding"
    def simple_Recommend_RSU(self, task):
        # Baseline: execute on the source RSU (no forwarding)
        self.pendingList.append((task.id, self.taskCounter))
        self.taskCounter += 1
        task.selected_RSU = task.original_RSU # the approach uses the source RSU as the execution RSU
        return task.selected_RSU

    def greedy_Recommend_RSU(self, task):
        # Greedy baseline: choose RSU based on a weighted combination of load and forwarding delay
        self.pendingList.append((task.id, self.taskCounter))
        self.taskCounter += 1
        candidate_rsus = task.vehicle.rsu_subgraph
        best_rsu = None
        best_score = float('inf')

        # Normalize load by maximum taskCounter among candidates
        max_counter = max(
            [self.env_state.RSU_and_Vehicle.get_rsu_by_id(r).taskCounter for r in candidate_rsus]
        ) or 1

        # Precompute and normalize delay by maximum delay among candidates
        max_delay = 1
        delays = {}
        for rsu_id in candidate_rsus:
            if rsu_id == task.original_RSU.rsu_id:
                delay = 0
            else:
                delay = self.env_state.calculate_e2e_delay(
                    task.original_RSU.rsu_id, rsu_id, task.task_size
                )
            delays[rsu_id] = delay
            if delay > max_delay:
                max_delay = delay

        # Weighted score: a * normalized_load + b * normalized_delay
        a = 0.6
        b = 0.4
        for rsu_id in candidate_rsus:
            rsu = self.env_state.RSU_and_Vehicle.get_rsu_by_id(rsu_id)
            counter = rsu.taskCounter
            norm_counter = counter / max_counter
            norm_delay = delays[rsu_id] / max_delay
            score = a * norm_counter + b * norm_delay
            if score < best_score:
                best_score = score
                best_rsu = rsu

        task.selected_RSU = best_rsu
        return best_rsu
        
    def simple_process_pendingList_and_log_result(self, this_episode):
        # Process pending tasks without RL updates (baseline modes)
        while self.pendingList:
            yield self.env.timeout(params.min_computation_demand)
            self.simple_add_train(this_episode)

        self.ep_reward_list.append(self.episodic_reward)
        self.ep_delay_list.append(self.episodic_delay)
        avg_reward = np.mean(self.ep_reward_list[-40:])
        avg_delay = np.mean(self.ep_delay_list[-40:])
        self.log_data.append((this_episode, avg_reward, self.episodic_reward, avg_delay,self.episodic_delay))
        print("No Global Model: Episode * {} * Avg Reward is ==> {}".format(this_episode, avg_reward), "This episode:", self.episodic_reward)
        self.avg_reward_list.append(avg_reward)
    
    def simple_add_train(self, this_episode):
        # Baseline: compute rewards but do not store transitions or train
        removeList = []
        for taskid, task_counter in self.pendingList:
            reward, delay = self.calcReward(taskid)
            if reward is not None:
                self.episodic_reward += reward
                self.episodic_delay += delay
                self.rewardsAll.append(reward)
                removeList.append((taskid, task_counter))

        for t in removeList:
            self.pendingList.remove(t)
            task = self.env_state.get_task_by_id(t[0])
            self.log_level1_data(task, this_episode)

    def log_level1_data(self, task,this_episode):
        # Global (Level-1) per-task record (non-flat mode)
        self.task_Assignments_info.append((
        this_episode,
        task.id,
        task.vehicle_id,
        task.original_RSU.rsu_id,
        task.submitted_time,
        task.selected_RSU.rsu_id,
        task.selected_rsu_start_time,
        task.delivered_RSU.rsu_id if task.delivered_RSU else "None",
        task.delivered_time if task.delivered_time else task.timeout_time,
        task.execution_status_flag,
        task.deadline_flag,
        task.final_status_flag
    ))        
    #--------------------------Flat arcitechture---------------------------
    def Recommend_action(self, task, this_episode):
        # Flat-mode: choose a single joint action (RSU, primary, backup, z)
        self.G_state = self.env_state.get_flat_state_for_unified_model(task)
        if self.taskCounter>1:
            tempx=list(self.tempbuffer[self.taskCounter-1])
            tempx[3]=self.G_state
            self.tempbuffer[self.taskCounter-1]=tuple(tempx)
            self.add_train(this_episode) 
        
        action_idx = self.agent.select_action(self.G_state, self.current_epsilon, use_softmax=True, temperature=1.5)
        rsu_id, primary_id, backup_id, z = self.action_list[action_idx]

        self.tempbuffer[self.taskCounter] = (self.G_state, action_idx, None, None)
        self.pendingList.append((task.id, self.taskCounter))
        self.taskCounter += 1

        # Convert ids to objects
        rsu_obj = self.env_state.RSU_and_Vehicle.RSUs[rsu_id]
        primary_server = self.env_state.get_server_by_id(primary_id)
        backup_server = self.env_state.get_server_by_id(backup_id)

        task.selected_RSU = rsu_obj
        task.selected_RSU.taskCounter += 1
        return rsu_obj, primary_server, backup_server, z

    def Set_Flat_setting(self):
        # ------------------------------------------------------------------
        # Build flat action space: all possible (RSU, primary, backup, z) tuples
        # ------------------------------------------------------------------
        def build_action_list():
            action_list = []
            rsu_dict = self.env_state.RSU_and_Vehicle.RSUs

            for rsu_id, rsu in rsu_dict.items():
                server_ids = list(rsu.serverIDs)  # includes edge + cloud

                # 1) z=0 => primary and backup may be the same
                for i in server_ids:
                    for j in server_ids:
                        action_list.append((rsu_id, i, j, 0))

                # 2) z=1 => parallel mode, unique pairs only (i < j)
                for i in server_ids:
                    for j in server_ids:
                        if i < j:
                            action_list.append((rsu_id, i, j, 1))

            print(f"[FlatGlobalModel] Total actions: {len(action_list)}")
            return action_list

        self.action_list = build_action_list()
        self.num_actions = len(self.action_list)

        # Flat-mode state dimension includes both RSU-level and server-level features
        num_rsus = len(self.env_state.RSU_and_Vehicle.RSUs)
        num_servers = len(self.env_state.servers)
        self.num_states = 5 * num_rsus + 9 * num_servers + num_rsus * (num_rsus - 1) + 3

    def log_flat_data (self, task, this_episode):
        original_rsu_id   = task.original_RSU.rsu_id if task.original_RSU else "None"
        selected_rsu_id   = task.selected_RSU.rsu_id if getattr(task, "selected_RSU", None) else "None"
        final_rsu_id      = task.delivered_RSU.rsu_id if getattr(task, "delivered_RSU", None) else "None"
        finished_time     = task.delivered_time if task.delivered_time is not None else task.timeout_time

        primary_id        = task.primaryNode.server_id if getattr(task, "primaryNode", None) else "None"
        backup_id         = task.backupNode.server_id if getattr(task, "backupNode", None) else "None"

        # This is the merged TaskAssignments record (global + local):
        self.task_Assignments_info.append((
            this_episode,           # episode
            task.id,                # task_id
            task.vehicle_id,        # vehicle_id

            original_rsu_id,        # original_rsu
            task.submitted_time,    # submitted_time

            selected_rsu_id,        # selected_RSU
            task.selected_rsu_start_time,  # start_executing
            final_rsu_id,           # final_rsu
            finished_time,          # finished_time

            primary_id,             # primary
            task.primaryStarted,    # primary_start
            task.primaryFinished,   # primary_end
            task.primaryStat,       # primary_status

            backup_id,              # backup
            task.backupStarted,     # backup_start
            task.backupFinished,    # backup_end
            task.backupStat,        # backup_status

            task.z,                 # z
            task.execution_status_flag,  # executaion_status
            task.deadline_flag,     # deadline_flag
            task.final_status_flag  # final_status
        ))
