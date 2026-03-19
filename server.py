# server.py
import simpy

class Server:
    def __init__(self, env, server_type, server_id, RSU_id, processing_frequency, failure_rate):
        # SimPy environment used for discrete-event simulation
        self.env = env

        # Server type label (e.g., "Edge" or "Cloud")
        self.server_type = server_type

        # Unique server identifier (used across the simulation)
        self.server_id = server_id

        # Parent RSU identifier (for edge servers; may also be used as metadata for cloud)
        self.RSU_id = RSU_id

        # Single-core server modeled as a priority queue (capacity=1)
        # priority is used in Task.backup() to allow retry tasks to preempt (lower number = higher priority)
        self.queue = simpy.PriorityResource(env, capacity=1)

        # Processing frequency of the server (used to compute service time: computation_demand / processing_frequency)
        self.processing_frequency = processing_frequency  # f_n(t) in the paper

        # Base failure rate of the server (used to compute fault probability during execution)
        self.failure_rate = failure_rate  # λ_n(t) in the paper

    def reset_queue(self, env):
        """Re-bind the server queue to a new SimPy environment (called at episode reset)."""
        self.env = env
        # Reinitialize the queue with the new environment to avoid carrying over old events/resources
        self.queue = simpy.PriorityResource(env, capacity=1)
