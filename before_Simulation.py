# before_Simulation.py
# Pre-simulation script:
# 1) Build SUMO-based graph (nodes/edges) + RSUs + vehicles/routes
# 2) Sample RSU-to-RSU link properties (failure rate, bandwidth, distance)
# 3) Save the extracted graph and metadata to graph_data.json
# 4) Generate a merged task queue (taskQueue.json) from per-vehicle task times
# 5) (Optional) Visualize the graph and RSU coverage
# 6) Generate Excel parameter files for servers and tasks

from Sumo_Graph import GraphNetwork
import generate_server_and_task_parameters

# Create the graph/network object
network = GraphNetwork()

# Parse SUMO files:
# - test.net.xml : road network (junctions/nodes + edges)
# - test.add.xml : RSU positions (POIs) and RSU coverage generation
# - test.rou.xml : vehicle routes and per-vehicle task schedules
network.load_sumo_data("SUMO/test.net.xml", "SUMO/test.add.xml", "SUMO/test.rou.xml")

# Randomly initialize RSU-to-RSU link properties used later in simulation:
# - failure rate per RSU pair
# - bandwidth per RSU pair
# - distance per RSU pair
network.set_RSU_failure_rate_bandwidths_distances()

# Persist the full graph + RSU info + vehicle info into a JSON file (graph_data.json)
network.save_graph()

# Generate the global task queue (merged across vehicles) and save to taskQueue.json
network.generate_Task_Queue()

# Plot the graph and RSU coverage circles for visualization/debugging
network.plot_graph()

# Generate:
# - homogeneous_server_info.xlsx
# - heterogeneous_server_info.xlsx
# - task_parameters.xlsx
generate_server_and_task_parameters.main()
