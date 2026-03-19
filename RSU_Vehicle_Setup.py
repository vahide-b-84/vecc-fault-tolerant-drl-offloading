# RSU_Vehicle_setup.py
from vehicles import Vehicle
from RSU import RSU
from configuration import parameters
import traci
import csv

class RSU_and_Vehicle_setup:
    def __init__(self, simpyEnv, graph_network):
        self.env = simpyEnv
        self.graph_network = graph_network
        
        # Vehicles (stored in a dict for fast O(1) lookup)
        self.num_vehicles = len(graph_network.vehicles)
        self.vehicles = {}

        # RSUs (stored in a dict for fast O(1) lookup)
        self.num_RSUs = len(graph_network.rsus)
        self.RSUs = {}

        # Initialize RSUs and vehicles
        self.setup_RSUs()
        self.setup_vehicles()
    
    def setup_RSUs(self):
        # Create RSU objects based on graph_network description
        rsu_data = self.graph_network.rsus

        for rsu_id, rsu_info in rsu_data.items():
            edge_numbers = rsu_info.get('edge_server_numbers', 0) 
            RSU_position = rsu_info.get('position', (0, 0)) 

            # Create RSU object and store it in dictionary
            self.RSUs[rsu_id] = RSU(self, rsu_id, edge_numbers, RSU_position, self.env)

            # Update global count of edge servers
            parameters.NUM_EDGE_SERVERS += edge_numbers

        print(f"{len(self.RSUs)} RSUs have been created! with total edge server:{parameters.NUM_EDGE_SERVERS}")  
    
    def get_rsu_by_id(self, rid):
        # Return RSU object by ID (O(1) lookup)
        return self.RSUs.get(rid, None)

    def setup_vehicles(self):
        # Create Vehicle objects for all vehicles in the graph
        vehicle_data = self.graph_network.vehicles
        for vehicle_id in vehicle_data.keys():
            self.vehicles[vehicle_id] = Vehicle(self, vehicle_id, self.env, self.graph_network)

    def get_vehicle_by_id(self, vid):
        # Return Vehicle object by ID (O(1) lookup)
        return self.vehicles.get(vid, None)
    
    def Start_SUMO(self, use_gui=True):
        # Open CSV file to store vehicle–RSU association over time
        with open("veh_rsu_coverage.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "VehicleID", "RSU_ID"])
            
            print("simulate_vehicle_movement started")

            # Select SUMO binary (GUI or command-line)
            sumo_binary = "sumo-gui" if use_gui else "sumo"

            # Start SUMO via TraCI
            traci.start([
                sumo_binary,
                "-c", "SUMO/test.sumocfg",
                "--quit-on-end",
                "--start",
                "--no-step-log"
            ])
            
            # Get SUMO simulation step length (usually 1 second)
            step_length = traci.simulation.getDeltaT()

            # Run until SUMO simulation ends
            while True:
                traci.simulationStep()
                yield self.env.timeout(step_length)

                # Update RSU association for each active vehicle
                for veh_id in traci.vehicle.getIDList():
                    vehicle = self.get_vehicle_by_id(veh_id)
                    if vehicle:
                        vehicle.set_current_rsu(writer)
    
    def extract_rsu_logs_and_assignments(self):
        # Collect per-RSU logs and task assignment records
        rsu_logs = {}
        rsu_assignments = {}

        for rsu_id, rsu in self.RSUs.items():
            rsu_logs[rsu_id] = rsu.log_data
            rsu_assignments[rsu_id] = rsu.task_Assignments_info

        return rsu_logs, rsu_assignments
