# vehicle.py
import random
from params import params
import math
#import logging
import traci  # Importing TraCI for SUMO interactions (vehicle position, simulation time, etc.)

'''logger = logging.getLogger(__name__)'''

class Vehicle:
    def __init__(self, rsu_and_vehicle, vehicle_id, env, graph_network):
        self.vehicle_id = vehicle_id
        self.rsu_and_vehicle = rsu_and_vehicle
        self.env = env
        self.graph_network = graph_network
 
        # Initial speed is read from pre-extracted graph_data (SUMO-derived)
        self.speed = self.graph_network.vehicles[self.vehicle_id]["speed"]  # Get initial speed from graph_data

        # Current RSU association (updated continuously during the simulation)
        self.Current_RSU = None

        # Current SUMO position (x, y)
        self.position = None

        # True RSU subgraph for this vehicle (ground-truth candidate RSUs along its trajectory)
        self.rsu_subgraph_true = list(self.graph_network.vehicles[self.vehicle_id]["rsu_subgraph"])

        # Predicted RSU subgraph (may be noisy under trajectory_noise scenario)
        self.rsu_subgraph_pred = list(self.rsu_subgraph_true)

        # Active subgraph used by the system (default: predicted)
        self.rsu_subgraph = self.rsu_subgraph_pred

        # Tasks that have been submitted by this vehicle and are waiting for results
        self.pending_tasks = []  # List of pending tasks
      
        #logger.info(f"Vehicle {self.vehicle_id} initialized.")
        

    def add_pending_task(self, task):
        """Add a task to the pending task list."""
        # Vehicle-side deadline heuristic (used to decide timeout vs success in request_results)
        task.deadline = task.submitted_time + 2.0 * (task.computation_demand / 10 + task.task_size / 20)  # or any preferred heuristic/formula

        self.pending_tasks.append(task)
        #logger.info(f"Task {task.id} added to pending tasks of {self.vehicle_id}")


    def request_results(self):
        # Vehicle checks whether tasks have timed out or have been cached at the current RSU
        
        for task in self.pending_tasks[:]:  # iterate over a copy because we may remove elements
            
            # Deadline missed: mark as failed for global-level evaluation
            if self.env.now > task.deadline:
                task.timeout_time = self.env.now  
                task.deadline_flag = 'F'
                self.pending_tasks.remove(task)
                #logger.warning(f"Deadline missed for Task {task.id}, Vehicle {self.vehicle_id} ... Marked as Failed")

            # If vehicle is currently connected to an RSU, try to fetch cached results
            elif self.Current_RSU:
                if task.id in self.Current_RSU.cached_results:
                    task.delivered_time = self.env.now
                    task.deadline_flag = 'S'
                    task.delivered_RSU = self.Current_RSU

                    # Remove the cached result once it is delivered to the vehicle
                    del self.Current_RSU.cached_results[task.id]

                    self.pending_tasks.remove(task)
                    #logger.info(f"Vehicle {self.vehicle_id} received result for task {task.id}")
                #else:
                    #logger.info(f"Vehicle {self.vehicle_id} not found Task {task.id} in {self.Current_RSU.rsu_id} at {self.env.now}")



    def set_current_rsu(self, writer):
        """Find the closest RSU to the vehicle using SUMO's real-time position."""
        
        # Query real-time vehicle position from SUMO via TraCI
        self.position = traci.vehicle.getPosition(self.vehicle_id)
        x, y = self.position

        # Find the closest RSU within communication range
        min_distance = float('inf')
        closest_rsu = None

        for rsu_id, rsu in self.graph_network.rsus.items():
            rsu_x, rsu_y = rsu["position"]
            distance = math.hypot(x - rsu_x, y - rsu_y)  

            # Candidate RSU must be within range, and closer than previously found RSUs
            if distance <= rsu["range"] and distance < min_distance:
                min_distance = distance
                closest_rsu = rsu_id

        # Update current RSU object (may be None if no RSU is in range)
        self.Current_RSU = self.rsu_and_vehicle.get_rsu_by_id(closest_rsu)

        # Optional logging (commented out)
        #if closest_rsu:
        #    logger.info(f"Vehicle {self.vehicle_id} is in range of {closest_rsu} at Simpytime {self.env.now} and SUMO time {traci.simulation.getTime()}")
        #else:
        #    logger.info(f"Vehicle {self.vehicle_id} is not in range of any RSU at Simpytime {self.env.now} and SUMO time {traci.simulation.getTime()}")
        
        # Log vehicle-RSU association over time (SUMO time is used here)
        writer.writerow([traci.simulation.getTime(), self.vehicle_id, closest_rsu])  # Log vehicle-RSU association

        # After RSU update, attempt to pull any completed results
        self.request_results()
        
    def apply_path_prediction_noise(self, p, all_rsu_ids, seed=None):
        # Apply noise to the predicted RSU subgraph (trajectory prediction uncertainty)
        rng = random.Random(seed)

        true_set = set(self.rsu_subgraph_true)
        all_set = set(all_rsu_ids)

        # FN (false negatives): drop some true RSUs from the predicted set
        pred = {r for r in true_set if rng.random() > p}

        # FP (false positives): add some non-true RSUs (scaled by the true path length)
        non_true = list(all_set - true_set)
        fp_count = int(round(p * len(true_set)))
        if fp_count > 0 and non_true:
            fp_count = min(fp_count, len(non_true))
            pred.update(rng.sample(non_true, fp_count))

        # Update predicted and active subgraph
        self.rsu_subgraph_pred = list(pred)
        self.rsu_subgraph = self.rsu_subgraph_pred

    def reset(self, new_SimPy_env, this_episode):
        """Reset the vehicle's state for a new simulation run."""
        self.env = new_SimPy_env
        self.Current_RSU = None
        self.pending_tasks.clear()
        #logger.info(f"Vehicle {self.vehicle_id} reset for new simulation run.")

        # Scenario-dependent update: apply trajectory noise to predicted RSU subgraph
        p = params.trajectory_noise_p
        seed = this_episode
        if params.scenario == "trajectory_noise":
            all_rsu_ids = list(self.graph_network.rsus.keys())
            self.apply_path_prediction_noise(p, all_rsu_ids, seed=seed)
