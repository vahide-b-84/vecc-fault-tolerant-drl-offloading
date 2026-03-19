# task.py
from params import params
import math
import random
class Task:

    def __init__(self, env, state, id, Vehicles, params_file):
        self.env = env
        self.env_state = state
        self.id = id
        
        # Execution placement (selected by orchestration logic)
        self.primaryNode = None
        self.backupNode = None
        self.z = None  # redundancy mode: 0=sequential, 1=parallel

        # Primary execution timeline/status
        self.primaryStarted = None
        self.primaryFinished = None
        self.primaryStat = None  # "success" or "failure"
        
        #self.primary_next_failure = None
        self.primary_service_time = None  # service time on primary server

        # Backup execution timeline/status
        self.backupStarted = None
        self.backupFinished = None
        self.backupStat = None  # "success" or "failure"        

        # Read task parameters from the params file (DataFrame)
        task_info_df = params_file

        # Find the row corresponding to this Task_ID
        task_row = task_info_df.loc[task_info_df['Task_ID'] == self.id]

        # Task parameters (used for transmission and computation timing)
        self.task_size = task_row['Task_Size'].values[0]
        self.computation_demand = task_row['Computation_Demand'].values[0]
        self.vehicle_id = task_row['Vehicle_ID'].values[0]
        self.vehicle= Vehicles.get(self.vehicle_id, None)  # Direct lookup (O(1))
        
               
        # End-to-end metadata across RSUs (used by Level-1/global evaluation)
        self.original_RSU= None  # RSU that receives the task from the vehicle
        self.submitted_time = None  # time the task is submitted at the original RSU
        
        self.selected_RSU=None  # RSU selected by the global model to execute the task
        self.selected_rsu_start_time=None  # time the task arrives at the selected RSU

        self.delivered_RSU=None  # RSU that delivers the result to the vehicle
        self.delivered_time = None  # time the result is delivered to the vehicle at its destination RSU
        self.timeout_time =None  # time when the vehicle marks a deadline miss

        # Status flags
        self.execution_status_flag = 'n'  # server-side status: 's' (success) / 'f' (failure) / 'n' (not finished)
        self.deadline_flag = 'N' # global view: 'S' (met deadline) / 'F' (missed) / 'N' (not resolved yet)
        self.final_status_flag = None
        self.deadline = None  # vehicle-side deadline (set by Vehicle)

        # Decision deadline used in sequential mode (computed after primary finishes)
        self.teta = None  
        
    def execute_task(self, X, Y, Z):
        # Set execution decision: primary server X, backup server Y, redundancy mode Z
        self.primaryNode=X
        self.backupNode=Y
        self.z = Z
        
        if self.z == 0:
            # Sequential strategy: run primary first; run backup only if primary fails
            self.primaryStarted = self.env.now
            yield self.env.process(self.primary())
            # teta: wait until the decision deadline before starting backup (if needed)
            if self.primaryStat == "failure":

                yield self.env.timeout(max(self.teta - (self.primaryFinished - self.primaryStarted), 0))
                self.backupStarted = self.env.now
                yield self.env.process(self.backup())
            
        else: ## z==1
            # Parallel strategy: run primary and backup concurrently
            self.primaryStarted = self.backupStarted = self.env.now
            processes = [
                self.env.process(self.primary()),
                self.env.process(self.backup())
            ]
            yield self.env.all_of(processes)
            
        # Final failure only if both primary and backup fail
        if self.primaryStat == "failure" and self.backupStat=="failure":
            self.execution_status_flag='f'

        # Forward task result to destination RSU(s) for vehicle pickup
        self.selected_RSU.forward_result(self) #forward task result to destination/s
        
    def primary(self):
        # Compute transmission delays (cloud has non-zero delays; edge is assumed local)
        inpDelay , outDelay = self.calc_input_output_delay(self.primaryNode)

        # Input transfer delay
        yield self.env.timeout(inpDelay)

        # Queueing on the server
        Q_time= self.env.now
        with self.primaryNode.queue.request(priority=1) as req:
            yield req  # Queueing time in server
            Q_time= self.env.now - Q_time

            # Register the assignment in EnvironmentState
            self.env_state.assign_task_to_server(self.primaryNode.server_id, self, "primary")# assign a task to a server as primary run

            # Calculate service time on primaryNode
            self.primary_service_time = self.computation_demand / self.primaryNode.processing_frequency

            # Failure rate may increase with load (queue length) and is capped
            failure_rate_adjusted=self.set_failure_rate(self.primaryNode)

            # Execute service time
            yield self.env.timeout(self.primary_service_time)
            
        # Sample failure using exponential model: P(fail) = 1 - exp(-lambda * service_time)
        fault_prob= 1-math.exp(-failure_rate_adjusted * self.primary_service_time)
        r=random.uniform(0, 1)
        if(r<fault_prob):
            self.primaryStat = "failure"
            
        else:
            # Output transfer delay (cloud only)
            yield self.env.timeout(outDelay)
            self.primaryStat = "success"
            self.execution_status_flag ='s'

        self.primaryFinished = self.env.now
        
        # Notify EnvironmentState that primary execution is complete
        self.env_state.complete_task(self.primaryNode.server_id, self, 'primary', self.primary_service_time)
        
        # Compute teta (decision deadline heuristic for sequential backup)
        self.teta= 1.5 * (self.primary_service_time + inpDelay + outDelay + Q_time) 

    def backup(self):

        inpDelay , outDelay = self.calc_input_output_delay(self.backupNode)

        # Use PriorityRequest if backupNode is the same as primaryNode
        if self.backupNode == self.primaryNode: # Retry sterategy
            # no inpDelay (retry on same server)
            with self.backupNode.queue.request(priority=0) as req: #high priority
                yield req  
                self.env_state.assign_task_to_server(self.backupNode.server_id, self, "backup") 
                backup_service_time = self.primary_service_time # as primary
                failure_rate_adjusted=self.set_failure_rate(self.backupNode)
                yield self.env.timeout(backup_service_time)

        else: # recovery block or first result strategy
            # Input transfer delay if backup is on a different server (cloud)
            yield self.env.timeout(inpDelay)
            with self.backupNode.queue.request(priority=1) as req:
                yield req 
                self.env_state.assign_task_to_server(self.backupNode.server_id, self, "backup") 
                backup_service_time = self.computation_demand / self.backupNode.processing_frequency # may differ from primary according to frequency of backup server
                failure_rate_adjusted=self.set_failure_rate(self.backupNode)
                yield self.env.timeout(backup_service_time)

        fault_prob= 1-math.exp(-failure_rate_adjusted * backup_service_time)
        r=random.uniform(0, 1)
        if(r<fault_prob):
            self.backupStat = "failure"
        else:
            yield self.env.timeout(outDelay)
            self.backupStat = "success"
            self.execution_status_flag="s"
         
        self.backupFinished = self.env.now
        
        # Notify EnvironmentState that backup execution is complete
        self.env_state.complete_task(self.backupNode.server_id, self, "backup", backup_service_time)

    def calc_input_output_delay(self, server_object):
        # Transmission delay model between RSU domain and server:
        # - Edge: assumed local => 0 delay
        # - Cloud: delay depends on rsu_to_cloud_bandwidth
        if server_object.server_type == "Edge":
            # Calculate input delay for Edge
            inpDelay = 0
        else:
            # Calculate input delay for Cloud
            inpDelay = self.task_size / params.rsu_to_cloud_bandwidth 

        # Output delay is the same as input delay
        outDelay = inpDelay   
        return inpDelay, outDelay
       
    def set_failure_rate(self, server_object):
            # Load-dependent failure rate adjustment:
            # adjusted = base_failure_rate + alpha[0] * queue_length, capped by alpha[1]
            if (server_object.server_type=="Edge"):
                failure_rate_adjusted = server_object.failure_rate + params.alpha_edge[0] * len(server_object.queue.queue)
                if failure_rate_adjusted>params.alpha_edge[1]:
                    failure_rate_adjusted=params.alpha_edge[1]

            else:
                failure_rate_adjusted = server_object.failure_rate + params.alpha_cloud[0] * len(server_object.queue.queue)
                if failure_rate_adjusted>params.alpha_cloud[1]:
                    failure_rate_adjusted=params.alpha_cloud[1]

            return failure_rate_adjusted
