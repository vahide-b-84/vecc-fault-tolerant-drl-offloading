# RSU.py: Level-2 model (Local RSU controller)
import numpy as np
import math
import torch
from DQN_template import DQNAgent
from PPO_template import PPOAgent
from params import params
import logging

logger = logging.getLogger(__name__)

class RSU:
    def __init__(self, rsu_and_vehicle, rsu_id, Edge_number, RSU_position, env):
        self.env = env
        self.env_state = None  # Will be assigned after EnvironmentState is created
        self.rsu_id = rsu_id
        self.rsu_position = RSU_position

        # Total servers visible to this RSU = edge servers + cloud servers
        self.serverNo = Edge_number + params.NUM_CLOUD_SERVERS
        self.serverIDs = []  # List of server IDs (edge + cloud) accessible from this RSU
        self.rsu_and_vehicle = rsu_and_vehicle

        # Local state/action dimensions (must match EnvironmentState.get_state())
        # State is typically built from per-server features + 2 RSU-level features
        self.num_states = 4 * self.serverNo + 2

        # Action space size:
        # - z=0: all ordered pairs (primary, backup) => serverNo * serverNo
        #        (primary==backup allowed: "retry" strategy)
        # - z=1: unique unordered pairs (i<j)       => serverNo*(serverNo-1)/2
        #        (parallel strategy with distinct servers)
        self.num_actions = (self.serverNo * self.serverNo) + (self.serverNo * (self.serverNo - 1)) // 2

        # Epsilon schedule used by local action selection (mostly for DQN exploration)
        self.epsilon_start = params.Local['epsilon_start']
        self.epsilon_end = params.Local['epsilon_end']
        self.epsilon_decay = params.Local['epsilon_decay']

        # Current decision state/action (updated per incoming task)
        self.state = []
        self.action = None

        # tempbuffer[taskCounter] = (s, a, r, s_next_placeholder)
        # Reward (r) and next-state (s') are filled later when the task outcome is known
        self.tempbuffer = {}
        self.taskCounter = 0
        self.pendingList = []  # List of (task_id, taskCounter) awaiting reward resolution

        # Episode-level metrics and logs
        self.rewardsAll = []
        self.ep_reward_list = []
        self.ep_delay_list = []
        self.avg_reward_list = []
        self.episodic_reward = 0
        self.episodic_delay = 0
        self.log_data = []
        self.task_Assignments_info = []

        # Action lookup list: index -> (primary_id, backup_id, z)
        # Must be built once after serverIDs are known (via generate_combinations)
        self.index_of_actions = []

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if params.model_summary == "ppo":
            # Local PPO agent (on-policy)
            # Transitions are stored during the episode; policy update happens at end of episode
            self.local_model = PPOAgent(
                num_states=self.num_states,
                num_actions=self.num_actions,
                hidden_layers=params.Local['hidden_layers'],
                device=device,
                gamma=params.Local['gamma'],
                actor_lr=params.Local['actor_lr'],
                critic_lr=params.Local['critic_lr'],
                clip_eps=params.Local['clip_eps'],
                k_epochs=params.Local['k_epochs'],
                batch_size=params.Local['batch_size'],
                entropy_coef=params.Local['entropy_coef'],
                reward_scale=params.Local.get('reward_scale', 1.0),
                gae_lambda=params.Local.get('gae_lambda', 0.95),
                value_loss_coef=params.Local.get('value_loss_coef', 0.5),
                max_grad_norm=params.Local.get('max_grad_norm', 0.5),
                activation=params.Local['activation'],
            )
            print(f"{self.rsu_id}: using PPOAgent as local model")
        else: # level-2:"dqn" ----> when level-1: dqn/greedy/NoForwarding
            # Local DQN agent (off-policy)
            # Transitions are stored in replay buffer; training happens online during the episode
            self.local_model = DQNAgent(
                num_states=self.num_states,
                num_actions=self.num_actions,
                hidden_layers=params.Local['hidden_layers'],
                device=device,
                gamma=params.Local['gamma'],
                lr=params.Local['lr'],
                tau=params.Local['tau'],
                buffer_size=params.Local['buffer_capacity'],
                batch_size=params.Local['batch_size'],
                activation=params.Local['activation'],
            )
            print(f"{self.rsu_id}: using DQNAgent as local model")

        # Cached completed results at this RSU (vehicle will pull from nearest RSU)
        self.cached_results = {}

    def get_epsilon(self, episode):
            # Linear epsilon decay (instead of exponential)
            epsilon = max(self.epsilon_end, self.epsilon_start - (episode / self.epsilon_decay))
            return epsilon
    
    def generate_combinations(self):
        # Build action index lookup:
        # 1) z=0 => all ordered (primary, backup) pairs (including primary==backup)
        for i in self.serverIDs:
            for j in self.serverIDs:
                self.index_of_actions.append((i, j, 0))

        # 2) z=1 => unique pairs (i<j) for parallel execution (avoid duplicates)
        for i in self.serverIDs:
            for j in self.serverIDs:
                if i < j:
                    self.index_of_actions.append((i, j, 1))

    def extract_parameters(self):
        # Decode selected action index into (primary_server_obj, backup_server_obj, z)
        index = self.action
        
        primary, backup, z = self.index_of_actions[index]
        return self.env_state.get_server_by_id(primary), self.env_state.get_server_by_id(backup), z

    def Recommend_XYZ(self, task, episode):
        # Called when a task has been assigned to this RSU for local execution decision
        self.taskCounter += 1 
        self.state = self.env_state.get_state(task, self.rsu_id)

        # For the previous task decision, fill in next-state (s') and train on any resolved rewards
        if self.taskCounter > 1:
            temp = list(self.tempbuffer[self.taskCounter - 1])
            temp[3] = self.state
            self.tempbuffer[self.taskCounter - 1] = tuple(temp)
            self.add_train(episode)

        epsilon = self.get_epsilon(episode)
        self.action = self.local_model.select_action(self.state, epsilon)

        # Convert action index -> (X=primary server, Y=backup server, Z=redundancy mode)
        X, Y, Z = self.extract_parameters()

        # Store pending transition; reward will be computed later when task outcome is known
        self.tempbuffer[self.taskCounter] = (self.state, self.action, None, [])
        self.pendingList.append((task.id, self.taskCounter))

        return X, Y, Z

    def calcReward(self, taskID):
        # Compute local reward once the task has finished (or failed)
        # Reward is based on execution delay and success/failure outcome
        task = self.env_state.get_task_by_id(taskID)
        z = task.z
        primaryStat = task.primaryStat
        backupStat = task.backupStat
        primaryFinished = task.primaryFinished
        primaryStarted = task.primaryStarted
        backupFinished = task.backupFinished
        backupStarted = task.backupStarted

        flag = "s"  # s=success, f=failure, n=not resolved yet
        delay = None

        if z == 0:
            # Sequential mode: backup runs only if primary fails
            if primaryStat == 'success' and backupStat is None and primaryFinished is not None:
                delay = primaryFinished - primaryStarted
            elif primaryStat == 'failure' and backupStat == 'success' and backupFinished is not None:
                delay = backupFinished - primaryStarted
            elif primaryStat == 'failure' and backupStat == 'failure':
                delay = backupFinished - primaryStarted
                flag = "f"
            else:
                flag = "n"
        else:
            # Parallel mode: primary and backup run concurrently
            if primaryStat == 'success' and backupStat == 'success' and primaryFinished is not None and backupFinished is not None:
                delay = min(primaryFinished, backupFinished) - primaryStarted
            elif primaryStat == 'success' and backupStat == 'failure' and primaryFinished is not None:
                delay = primaryFinished - primaryStarted
            elif primaryStat == 'failure' and backupStat == 'success' and backupFinished is not None:
                delay = backupFinished - backupStarted
            elif primaryStat == 'failure' and backupStat == 'failure':
                delay = max(backupFinished - backupStarted, primaryFinished - primaryStarted)
                flag = "f"
            elif primaryStat == 'success' and backupStat is None and primaryFinished is not None:
                delay = primaryFinished - primaryStarted
            elif primaryStat is None and backupStat == 'success' and backupFinished is not None:
                delay = backupFinished - backupStarted
            else:
                flag = "n"

        # Reward shaping constants (Level-2)
        success_reward_weight = 1.0
        failure_penalty_weight = 20.0
        max_success_reward = 30.0
        min_failure_penalty = -3.0
        max_failure_penalty = -3 * max_success_reward  # i.e., -90.0

        if flag == "f":
            # Negative reward proportional to delay (capped)
            reward = -failure_penalty_weight * delay
            reward = max(min(reward, min_failure_penalty), max_failure_penalty)

        elif flag == "s":
            # Positive reward decreases with higher delay (bounded)
            reward = success_reward_weight * (
                math.log(1 - (1 / math.exp(math.sqrt(delay)))) / math.log(0.995)
            )
            reward = min(reward, max_success_reward)        

        else:
            # Reward is not available yet (task still running or missing timestamps)
            reward = None
        
        return reward, delay

    def add_train(self, episode):
        # Try to resolve rewards for tasks in pendingList; store transitions and/or train accordingly
        removeList = []
        if params.model_summary == "ppo":
            # PPO: store transitions during the episode; update policy at end of episode
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

                    # On-policy: store into PPO rollout buffer (training happens later)
                    self.local_model.store_transition(s, a, r, s_, done=False)

                    removeList.append((taskid, task_counter))

        else:
            # DQN: off-policy replay + online training during the episode
            if len(self.local_model.replay_buffer) > 0:
                self.local_model.train_step()
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

                    # Store transition and train step
                    self.local_model.store_transition((s, a, r, s_))
                    self.local_model.train_step()

                    removeList.append((taskid, task_counter))

        # Remove resolved tasks from pendingList and write a compact per-task log record
        for t in removeList:
            self.pendingList.remove(t)
            task = self.env_state.get_task_by_id(t[0])
            self.log_level2_data(task,episode)


    def forward_result(self, task):
        # Forward the completed task result to all RSUs in the vehicle's RSU subgraph
        for rsuid in task.vehicle.rsu_subgraph:
            dist_rsu = self.rsu_and_vehicle.get_rsu_by_id(rsuid)
            self.env.process(dist_rsu.receive_result(task))

    def receive_task(self, task):
        # Simulate forwarding delay from original RSU to this RSU (if different)
        delay = 0 if task.original_RSU.rsu_id == self.rsu_id else self.env_state.calculate_e2e_delay(task.original_RSU.rsu_id, self.rsu_id, task.task_size)
        yield self.env.timeout(delay)

        # Timestamp of arrival at selected RSU (used by global logging/metrics)
        task.selected_rsu_start_time = self.env.now 
    
    def receive_result(self, task):
        # Result delivery delay from selected RSU to this RSU (simplified to 1 if different)
        #delay = 0 if task.selected_RSU.rsu_id == self.rsu_id else self.env_state.calculate_e2e_delay(task.selected_RSU.rsu_id, self.rsu_id, params.task_result_size)
        delay = 0 if task.selected_RSU.rsu_id == self.rsu_id else 1
        yield self.env.timeout(delay)

        # Cache result locally so that the vehicle can retrieve it when connected
        self.cached_results[task.id] = task
        #logger.info(f"                  {self.rsu_id} cached result of Task {task.id} at {self.env.now}")
        #self.env.process(self.remove_cached_result(task.id, params.task_timeout_caching))

    def remove_cached_result(self, task_id, timeout):
        # Optional TTL cleanup for cached results
        yield self.env.timeout(timeout)
        self.cached_results.pop(task_id, None)
        #logger.info(f"RSU {self.rsu_id} pop cached result of Task {task_id} at {self.env.now}")

    def process_pendingList_and_log_result(self, episode):
        # Finalize the last stored transition's next-state and drain pendingList
        if self.taskCounter > 0:
            temp = list(self.tempbuffer[self.taskCounter])
            temp[3] = self.state
            self.tempbuffer[self.taskCounter] = tuple(temp)

        # Keep trying until all pending tasks have resolved rewards
        while self.pendingList:
            yield self.env.timeout(params.min_computation_demand)
            self.add_train(episode)

        if params.model_summary == "ppo":
            # PPO on-policy update step (must be called after collecting rollout transitions)
            self.local_model.train_step()

        # Episode-level logging (moving average uses last 40 episodes)
        self.ep_reward_list.append(self.episodic_reward)
        self.ep_delay_list.append(self.episodic_delay)
        avg_r = np.mean(self.ep_reward_list[-40:])
        avg_d = np.mean(self.ep_delay_list[-40:])
        self.avg_reward_list.append(avg_r)
        self.log_data.append((episode, avg_r, self.episodic_reward, avg_d, self.episodic_delay))
        print(f"{self.rsu_id}: Episode {episode} | Avg R: {avg_r:.2f} | This Episode R: {self.episodic_reward:.2f}")
  
    def reset(self, new_env):
        # Reset RSU buffers for a new episode
        self.env = new_env
        self.episodic_reward = 0
        self.episodic_delay = 0
        self.tempbuffer.clear()
        self.taskCounter = 0
        self.pendingList.clear()
        self.cached_results.clear()

    def log_level2_data (self, task, episode):
        # Compact local (Level-2) per-task record for later analysis/CSV export
        self.task_Assignments_info.append((
        episode, task.id, task.vehicle_id,
        task.primaryNode.server_id, task.primaryStarted, task.primaryFinished,
        task.primaryStat, task.backupNode.server_id, task.backupStarted,
        task.backupFinished, task.backupStat, task.z, task.execution_status_flag
        ))

    @property
    def load(self):
        # Simple load proxy: number of unresolved (pending) tasks
        return len(self.pendingList)
