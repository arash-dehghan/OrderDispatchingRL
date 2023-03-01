import sys
sys.dont_write_bytecode = True
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import argparse
import pickle
from Environment import Environment
from CentralAgent import CentralAgent
from ValueFunction import DRL
from LearningAgent import LearningAgent
from Experience import Experience
from ResultCollector import ResultCollector
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def run_epoch(envt, central_agent, value_function, requests, request_generator, agents_predefined, is_training = False):
	# Initialize the start time for the day
	envt.current_time = envt.start_epoch
	# Get the number of decision epochs
	ts = int((envt.stop_epoch - envt.start_epoch) / envt.epoch_length)
	# Setup the environment
	Experience.envt = envt
	# Copy the predifined agents
	agents = deepcopy(agents_predefined)
	# Set the global order ID and total orders served to 0
	global_order_id, total_orders_served = 0, 0
	# Some other info to initialize at start of problem horizon (beginning of day)
	graph_seen, graph_served, times = ([] for _ in range(3))
	total_rewards = 0
	experiences = []
	store_info = {}

	# For each decision epoch throughout the day
	for t in range(ts):
		# Generate and add deadlines to new orders, add new orders to remaining orders
		# If we're training
		if is_training:
			# We generate new requests given the time and set their deadlines based on the deadlines allowed parameter
			current_orders = [central_agent.set_deadlines(order, i) for i, order in enumerate(request_generator.get_requests(envt.current_time), start=global_order_id)]
			# We also sort them from orders that have drop-offs closest to warehouse to longest
			current_orders = sorted(current_orders, key=lambda order: order.travel_duration)
		# Else if we're testing
		else:
			# Get the pre-set testing orders for the given day and time
			current_orders = [central_agent.set_deadlines(order, i) for i,order in enumerate(requests[envt.current_time], start=global_order_id)]
			# We also sort them from orders that have drop-offs closest to warehouse to longest
			current_orders = sorted(current_orders, key=lambda order: order.travel_duration)
		# Update the global order ID for the day
		global_order_id += len(current_orders)

		# If we're training and it is not the first decision epoch (we have a next_state we need to get from the previous decision epoch)
		if is_training and len(store_info) > 0:
			# Get the next order
			next_order = deepcopy(current_orders[0])
			# Create the next state based on that next order
			next_state = central_agent.get_state(agents, next_order, len(current_orders), 0, deepcopy(current_orders))
			# Append experience to experiences
			experiences.append(Experience(deepcopy(store_info['state']), deepcopy(store_info['action']), store_info['reward'], deepcopy(next_state), False))
			# Update replay buffer with all experiences from past decision epoch
			for experience in experiences:
				# Update replay buffer
				value_function.replay_buffer.push(value_function.replay_buffer.Transition(experience.state[0], experience.action, experience.reward, experience.next_state[0], experience.terminal))

		# print(f'Time: {envt.current_time} Minutes')
		# print(f'Agent 0 state at beginning of timestep: {agents[0]}')
		# print(f'Number of orders arrived: {len(current_orders)}')

		# Store experiences 
		experiences = []
		store_info = {}

		# Count orders served at this decision epoch
		orders_served = 0

		# For each order in the current batch of orders
		for order_index in range(len(current_orders)):
			# Get the order from the index
			order = deepcopy(current_orders[order_index])
			# Get the state of the order
			state = central_agent.get_state(agents, order, len(current_orders), order_index, deepcopy(current_orders))
			# Get the action to take based on the state (0 = accept, 1 = reject)
			action = value_function.get_action(state, is_training)
			# action = 0
			# Get the reward for the action at the state
			reward = central_agent.get_reward(agents, order, action)
			# If the order is the last in the current decision epoch
			if order_index == len(current_orders) - 1:
				if is_training:
					# If it is not the end of the day (i.e: no more orders)
					if (t != (ts - 1)):
						store_info = {'state': state, 'action': action, 'reward': reward, 'terminal': False}
					else:
						experiences.append(Experience(deepcopy(state), deepcopy(action), reward, deepcopy(state), True))
						store_info = {}
			# If the order is NOT the last in the current decision epoch
			else:
				if is_training:
					next_order = deepcopy(current_orders[order_index + 1])
					next_state = central_agent.get_state(agents, next_order, len(current_orders), order_index + 1, deepcopy(current_orders))
					experiences.append(Experience(deepcopy(state), deepcopy(action), reward, deepcopy(next_state), False))
			# Update the total number of orders served in the current decision epoch and also the total rewards for the day
			orders_served = (orders_served + 1) if reward > 0 else orders_served
			total_rewards += reward

		# print(f'Number of orders matched to: {orders_served}')

		# Update if training
		if is_training:
			if len(store_info) == 0:
				for experience in experiences:
					# Update replay buffer
					value_function.replay_buffer.push(value_function.replay_buffer.Transition(experience.state[0], experience.action, experience.reward, experience.next_state[0], experience.terminal))
			# Update value function every TRAINING_FREQUENCY timesteps
			if ((int(envt.current_time) / int(envt.epoch_length)) % 1 == 0):
				value_function.update(central_agent)
			if value_function.training_iterations % 1000 == 0:
				value_function.hard_update()

		# Set the new trajectories for each agent
		# print(f'Agent 0 state after matchings prior to movement: {agents[0]}')
		# print(f'Returns to warehouse in : {envt._get_ordering_return_time(deepcopy(agents[0].orders_to_pickup))} minutes')
		central_agent.set_new_paths(agents)
		total_orders_served += orders_served
		graph_seen.append(len(current_orders))
		graph_served.append(orders_served)
		times.append(envt.current_time)

		# Update the time
		envt.current_time += envt.epoch_length

		# print('===')

		# Update epsilon if training
		if is_training:
			value_function.epsilon = max(value_function.epsilon_min, value_function.epsilon * value_function.epsilon_decay)
			envt.num_days_trained += 1
			value_function.training_iterations += 1
	return np.array([graph_served, graph_seen, total_rewards])

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--numagents', type=int, default=10)
	parser.add_argument('-data', '--data', type=str, choices=['Bangalore', 'Chicago', 'Brooklyn', 'Iowa'], default='Brooklyn')
	parser.add_argument('-break_length', '--break_length', type=int , default=16)
	parser.add_argument('-variation_percentage', '--variation_percentage', type=float , default=0.0)
	parser.add_argument('-speed_var', '--speed_var', type=float , default=0.2)
	parser.add_argument('-num_locations', '--num_locations', type=int , default=100)
	parser.add_argument('-road_speed', '--road_speed', type=float, default=20.0) #km/h
	parser.add_argument('-epoch_length', '--epoch_length', type=int , default=5)
	parser.add_argument('-dt', '--delaytime', type=float, default=10)
	parser.add_argument('-vehicle_cap', '--capacity', type=int, default=1)
	parser.add_argument('-train_days', '--train_days', type=int, default=150)
	parser.add_argument('-test_days', '--test_days', type=int, default=20)
	parser.add_argument('-test_every', '--test_every', type=int, default=5)
	parser.add_argument('-seed', '--seed', type=int , default=1)
	parser.add_argument('-trainable', '--trainable', type=int, default=0)
	args = parser.parse_args()

	filename = f"{args.data}_{args.epoch_length}_{args.road_speed}_{args.num_locations}_{args.seed}"
	request_generator = pickle.load(open(f'../data/generations/{filename}/data_{filename}.pickle','rb'))
	envt = Environment(args.numagents, args.epoch_length, args.capacity, args.data, args.break_length)
	central_agent = CentralAgent(envt, args.numagents, args.delaytime, args.capacity)
	value_function = DRL(envt)

	test_data = request_generator.create_test_scenarios(args.test_days, args.epoch_length)
	test_shift_start_times = [[request_generator.get_shift_start_time() for _ in range(20)][:args.numagents] for _ in range(args.test_days)]
	stops = [i for i in range(args.test_every,args.train_days + args.test_every,args.test_every)]
	result_collector = ResultCollector()
	final_results = []
	all_results = {}
	i = 0

	# Initial myopic results
	tot_seen, tot_served, tot_rewards = 0, 0, 0
	for test_day in tqdm(range(args.test_days)):
		orders = test_data[test_day]
		agents = [LearningAgent(i, args.break_length, test_shift_start_times[test_day][i]) for i in range(args.numagents)]
		results = run_epoch(envt, central_agent, value_function, orders, request_generator, agents, False)
		tot_seen += sum(results[1])
		tot_served += sum(results[0])
		tot_rewards += results[2]
		# final_results.append(results)
	all_results[i] = (tot_served,tot_seen, tot_rewards)
	print(all_results)
	# exit()
	# result_collector.update_results(i, final_results)

	# Train the model
	for train_day in tqdm(range(args.train_days)):
		agents = [LearningAgent(i, args.break_length, request_generator.get_shift_start_time()) for i in range(args.numagents)]
		run_epoch(envt, central_agent, value_function, None, request_generator, agents, True)
		i += 1
		if i in stops:
			final_results = []
			# Get the test results
			tot_seen, tot_served, tot_rewards = 0, 0, 0
			for test_day in range(args.test_days):
				orders = test_data[test_day]
				agents = [LearningAgent(i, args.break_length, test_shift_start_times[test_day][i]) for i in range(args.numagents)]
				results = run_epoch(envt, central_agent, value_function, orders, request_generator, agents, False)
				tot_seen += sum(results[1])
				tot_served += sum(results[0])
				tot_rewards += results[2]
				# final_results.append(results)
			all_results[i] = (tot_served,tot_seen,tot_rewards)
			# result_collector.update_results(i, final_results)
			print(all_results)

	# print('===')
	# print(value_function.losses)

	# file_data = f'{args.data}_{args.numagents}_{args.break_length}_{args.delaytime}'
	# value_function.model.save(f'../Results/{file_data}.h5')
	# with open(f'../Results/{file_data}.pickle', 'wb') as handle:
	# 	pickle.dump(result_collector, handle, protocol=pickle.HIGHEST_PROTOCOL)


