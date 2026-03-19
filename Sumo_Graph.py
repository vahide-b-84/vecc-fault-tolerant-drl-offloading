#Sumo_Graph.py
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
import numpy as np
import sumolib
import json
import networkx as nx
from params import params
import math
from heapq import heappush, heappop
from collections import defaultdict

class GraphNetwork:
    def __init__(self):
        # These are set once SUMO data is loaded / parsed
        self.num_nodes = None  # set in load_graph() based on loaded nodes
        self.num_rsus =  None  # set in load_graph() based on loaded RSUs

        # Road network graph (junctions as nodes, roads as edges)
        self.G = nx.Graph()

        # Junction positions: {node_id: (x, y)}
        self.positions = {}  # filled in load_sumo_data()

        # RSUs: {rsu_id: {"position": [x, y], "range": r, "edge_server_numbers": n}}
        self.rsus = {}  # filled in extract_rsus_from_additional()

        # Vehicles: {vehicle_id: {"path": [...], "speed": v, "task_times": [...], "rsu_subgraph": [...]} }
        self.vehicles = {}  # filled in extract_vehicle_data()

        # Link-level attributes between RSU pairs (used by higher-level modules)
        self.RSU_Pairs_failure_rate = {}
        self.RSU_distances = {}
        self.RSU_link_bandwidths = {}

    # Load SUMO network + RSUs + vehicle routes/tasks into this graph structure
    def load_sumo_data(self, net_file, additional_file, route_file):
        # Load SUMO net (junctions and edges) using sumolib
        net = sumolib.net.readNet(net_file)

        # Add junctions (nodes) with coordinate attribute
        for junction in net.getNodes():
            node_id = junction.getID()
            node_pos = junction.getCoord()
            self.positions[node_id] = node_pos
            self.G.add_node(node_id, pos=node_pos)  # node attribute: pos=(x, y)

        # Add edges (roads) with length and SUMO edge id
        for edge in net.getEdges():
            edge_id = edge.getID()  # edge id as stored in SUMO net
            from_junction = edge.getFromNode().getID()
            to_junction = edge.getToNode().getID()
            edge_length = edge.getLength()
            self.G.add_edge(from_junction, to_junction, length=edge_length, id=edge_id)

        # RSUs from additional file (POIs)
        self.extract_rsus_from_additional(additional_file)

        # Vehicles, routes, and task-time generation from route file
        self.extract_vehicle_data(route_file)

    # Extract RSU placement/coverage/server-count from SUMO additional file (e.g., test.add.xml)
    def extract_rsus_from_additional(self, additional_file):
        tree = ET.parse(additional_file)
        root = tree.getroot()

        rsus = {}  # output format used by the rest of the simulation

        # Keep deterministic ordering by sorting POIs by their SUMO default id "poi_k"
        sorted_pois = sorted(root.findall("poi"), key=lambda poi: int(poi.get("id").split("_")[1]))

        for idx, poi in enumerate(sorted_pois):  # idx determines RSU_0, RSU_1, ...
            x = float(poi.get("x"))
            y = float(poi.get("y"))

            # Random RSU coverage radius within configured range
            range_radius = random.uniform(params.RSU_radius[0], params.RSU_radius[1])

            # Random number of edge servers attached to this RSU (within configured bounds)
            num_edge_servers = random.randint(params.RSUs_EDGE_SERVERS[0], params.RSUs_EDGE_SERVERS[1])

            # RSU IDs are normalized to RSU_{idx}
            rsu_id = f"RSU_{idx}"

            rsus[rsu_id] = {
                "position": [x, y],
                "range": range_radius,
                "edge_server_numbers": num_edge_servers
            }

        # Save RSUs in the class state
        self.rsus = rsus

    # Extract vehicle routes and generate per-vehicle task arrival times from SUMO routes file (e.g., test.rou.xml)
    def extract_vehicle_data(self, route_file):
        tree = ET.parse(route_file)
        root = tree.getroot()

        # Map each <route id="..."> to its list of edges
        routes_dict = {route.get("id"): route.get("edges").split() for route in root.findall("route")}

        vehicle_data = {}

        for vehicle in root.findall("vehicle"):
            vehicle_id = vehicle.get("id")
            route_id = vehicle.get("route")
            speed = float(vehicle.get("departSpeed", 0))

            # Validate route existence
            if route_id not in routes_dict:
                print(f"⚠ Warning: Route {route_id} not found for vehicle {vehicle_id}")
                continue

            path = routes_dict[route_id]

            # Generate task generation times (arrival process) for this vehicle
            task_arrival_rate = np.random.uniform(params.TASK_ARRIVAL_RATE_range[0], params.TASK_ARRIVAL_RATE_range[1])
            task_times = []
            t = 0
            for _ in range(params.Vehicle_taskno):
                inter_arrival_time = np.random.poisson(1 / task_arrival_rate)
                t += inter_arrival_time
                task_times.append(t)

            # RSU subgraph: all RSUs that cover at least one edge in the vehicle route
            rsu_subgraph = self.get_rsu_subgraph(path)

            vehicle_data[vehicle_id] = {
                "path": path,
                "speed": speed,
                "task_times": task_times,
                "rsu_subgraph": rsu_subgraph  # RSUs that can potentially serve this vehicle along its route
            }

        self.vehicles = vehicle_data

    # Compute set/list of RSUs that cover the vehicle's path (edge-based coverage test)
    def get_rsu_subgraph(self, path):
        rsu_subgraph = set()  # stores RSU ids that cover any edge in the path

        for edge_id in path:
            for rsu_id, rsu in self.rsus.items():
                rsu_position = rsu["position"]
                rsu_range = rsu["range"]
                if self.check_edge_rsu_coverage(edge_id, rsu_position, rsu_range):
                    rsu_subgraph.add(rsu_id)

        return list(rsu_subgraph)

    # Check if a SUMO edge (as a line segment between its nodes) intersects the RSU coverage circle
    def check_edge_rsu_coverage(self, edge_id, rsu_position, rsu_range):
        # Get edge endpoints (from/to nodes) from the SUMO net xml
        edge_data = self.get_edge_data_from_xml(edge_id)
        start_node = edge_data['from']
        end_node = edge_data['to']

        # Node positions in the NetworkX graph
        start_pos = self.G.nodes[start_node]['pos']
        end_pos = self.G.nodes[end_node]['pos']

        # RSU center (rx, ry)
        rx, ry = rsu_position

        # Quick check: if both endpoints are inside the circle => fully covered
        if (
            (start_pos[0] - rx)**2 + (start_pos[1] - ry)**2 <= rsu_range**2 and
            (end_pos[0] - rx)**2 + (end_pos[1] - ry)**2 <= rsu_range**2
        ):
            return True

        # Otherwise check line segment-circle intersection using quadratic discriminant
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]

        A = dx**2 + dy**2
        B = 2 * (dx * (start_pos[0] - rx) + dy * (start_pos[1] - ry))
        C = (start_pos[0] - rx)**2 + (start_pos[1] - ry)**2 - rsu_range**2

        discriminant = B**2 - 4 * A * C
        if discriminant < 0:
            return False  # no intersection

        sqrt_discriminant = math.sqrt(discriminant)
        t1 = (-B - sqrt_discriminant) / (2 * A)
        t2 = (-B + sqrt_discriminant) / (2 * A)

        # If any intersection parameter lies within the segment [0, 1] => covered
        if 0 <= t1 <= 1 or 0 <= t2 <= 1:
            return True

        return False

    # Helper: fetch SUMO edge endpoints (from/to nodes) by parsing the net xml file
    def get_edge_data_from_xml(self, edge_id):
        tree = ET.parse("SUMO/test.net.xml")
        root = tree.getroot()

        for edge in root.findall("edge"):
            if edge.get("id") == edge_id:
                from_node = edge.get("from")
                to_node = edge.get("to")
                return {"from": from_node, "to": to_node}

        raise ValueError(f"Edge {edge_id} not found in the network file.")

    # Precompute per-RSU-pair: failure rate, distance, and bandwidth (all symmetric)
    def set_RSU_failure_rate_bandwidths_distances(self):
        self.RSU_Pairs_failure_rate = {}
        self.RSU_distances = defaultdict(dict)
        self.RSU_link_bandwidths = {}

        rsu_ids = list(self.rsus.keys())
        for i, rsu_i in enumerate(rsu_ids):
            pos_i = self.rsus[rsu_i]["position"]
            for j, rsu_j in enumerate(rsu_ids):
                if i == j:
                    continue

                # Link failure rate between RSU_i and RSU_j (symmetric)
                fr = np.random.uniform(params.link_failure_rate_range[0],
                                       params.link_failure_rate_range[1])
                self.RSU_Pairs_failure_rate[(rsu_i, rsu_j)] = fr
                self.RSU_Pairs_failure_rate[(rsu_j, rsu_i)] = fr

                # Euclidean distance between RSU centers (symmetric)
                pos_j = self.rsus[rsu_j]["position"]
                d = math.hypot(pos_i[0]-pos_j[0], pos_i[1]-pos_j[1])
                self.RSU_distances[rsu_i][rsu_j] = d
                self.RSU_distances[rsu_j][rsu_i] = d

                # Link bandwidth between RSU_i and RSU_j (symmetric)
                bw = np.random.uniform(params.RSU_LINK_BANDWIDTH_RANGE[0],
                                       params.RSU_LINK_BANDWIDTH_RANGE[1])
                self.RSU_link_bandwidths[(rsu_i, rsu_j)] = bw
                self.RSU_link_bandwidths[(rsu_j, rsu_i)] = bw

    # Save the graph and all extracted/generated data to JSON for later reuse
    def save_graph(self):
        """Save the graph in JSON format, including vehicle data if provided."""
        graph_data = {
            "nodes": {node: {"pos": self.G.nodes[node]["pos"]} for node in self.G.nodes},

            # Save edges as tuples: (u, v, length, sumo_edge_id)
            "edges": [(u, v, self.G[u][v]["length"], self.G[u][v].get("id", "")) for u, v in self.G.edges],  # store edge ids too

            "rsus": self.rsus,

            # Serialize tuple keys as strings: "src->dst"
            "RSU_Pairs_failure_rate": {f"{a}->{b}": v for (a, b), v in self.RSU_Pairs_failure_rate.items()},
            "RSU_distances": {src: dsts for src, dsts in self.RSU_distances.items()},
            "RSU_link_bandwidths": {f"{a}->{b}": v for (a, b), v in self.RSU_link_bandwidths.items()},
        }

        # Add vehicles if already extracted/generated
        if self.vehicles:
            graph_data["vehicles"] = self.vehicles

        filename = "graph_data.json"
        with open(filename, "w") as f:
            json.dump(graph_data, f, indent=4)
        print(f"Graph saved to {filename}")
    
    # Load graph/RSU/vehicle/link attributes back from JSON file
    def load_graph(self, filename="graph_data.json"):
        """Load graph, RSUs, and vehicle data from JSON file."""
        with open(filename, "r") as f:
            data = json.load(f)

        # Reset current in-memory structures
        self.G.clear()
        self.positions.clear()
        self.rsus.clear()
        self.vehicles.clear()
        self.RSU_Pairs_failure_rate.clear()
        self.RSU_link_bandwidths.clear()
        self.RSU_distances.clear

        # Restore nodes and their positions
        for node, attrs in data["nodes"].items():
            self.G.add_node(node, pos=attrs["pos"])
            self.positions[node] = attrs["pos"]

        # Restore edges
        for u, v, length, edge_id in data["edges"]:
            self.G.add_edge(u, v, length=length, edge_id=edge_id)

        # Restore RSUs and vehicles
        self.rsus = data["rsus"]
        self.vehicles = data.get("vehicles", {})

        # Update counts from loaded content
        self.num_nodes = len(self.positions)
        self.num_rsus = len(self.rsus)

        # De-serialize string keys back to tuple keys
        self.RSU_Pairs_failure_rate = {
            tuple(k.split("->")): v for k, v in data.get("RSU_Pairs_failure_rate", {}).items()
        }
        self.RSU_distances = data.get("RSU_distances", {})
        self.RSU_link_bandwidths = {
            tuple(k.split("->")): v for k, v in data.get("RSU_link_bandwidths", {}).items()
        }
        print("Graph successfully loaded.")

    # Visualize the road graph and RSU coverage circles
    def plot_graph(self):
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        pos = {node: (data["pos"][0], data["pos"][1]) for node, data in self.G.nodes(data=True)}
        nx.draw(self.G, pos, with_labels=True, node_color='orange', edge_color='gray', node_size=50, font_size=8)
        
        # Draw RSU coverage as circles + RSU markers
        for rsu_id, rsu in self.rsus.items():
            x, y = rsu["position"]
            range_radius = rsu["range"]

            circle = plt.Circle((x, y), range_radius, color='blue', alpha=0.1, zorder=0)
            ax.add_patch(circle)

            plt.scatter(x, y, color='red', marker='^', s=100, zorder=1)
            plt.text(x, y + 10, rsu_id, color='black', fontsize=5, ha='center')

        # Legend
        legend_elements = [
            Line2D([0], [0], marker='^', color='w', label='RSU Node', markerfacecolor='red', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Regular Node', markerfacecolor='orange', markersize=10)
        ]
        plt.legend(handles=legend_elements, loc="upper right")
        
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Graph with RSU Coverage and Nodes")
        plt.grid(True)
        plt.show()

    # Build a global task scheduling list (merged across all vehicles) and save it as taskQueue.json
    def generate_Task_Queue(self):
        # Read graph data (including vehicles and their generated task_times)
        with open('graph_data.json', 'r') as file:
            data = json.load(file)

        # Merge per-vehicle sorted task time lists using a heap (priority queue)
        heap = []
        for vehicle_id, vehicle_info in data["vehicles"].items():
            task_times = vehicle_info["task_times"]
            if task_times:
                heappush(heap, (task_times[0], vehicle_id, 0))  # (task_time, vehicle_id, index_in_task_times)

        # Convert absolute times into interarrival times (global merged order)
        interarrival_list = []
        previous_time = 0
        while heap:
            current_time, vehicle_id, index = heappop(heap)
            interarrival_time = current_time - previous_time
            interarrival_list.append({
                "vehicle_id": vehicle_id,
                "time": current_time,
                "interarrival_time": interarrival_time
            })
            previous_time = current_time

            # Push next task from the same vehicle
            next_index = index + 1
            task_times = data["vehicles"][vehicle_id]["task_times"]
            if next_index < len(task_times):
                heappush(heap, (task_times[next_index], vehicle_id, next_index))

        # Optional debug print (kept as a triple-quoted block in original code)
        '''for item in interarrival_list:
            print(item)'''

        # Save merged queue
        output_file = 'taskQueue.json'
        with open(output_file, 'w') as outfile:
            json.dump(interarrival_list, outfile, indent=4)

        print(f"The task scheduling list has been saved in the file {output_file}.")
