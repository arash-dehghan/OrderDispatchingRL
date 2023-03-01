from copy import deepcopy
import time
import pickle
import numpy as np
from ReplayBuffer import SimpleReplayBuffer, PrioritizedReplayBuffer
from os.path import isfile, isdir

from RequestOrder import RequestOrder
from keras import Sequential
from keras.models import Model, load_model, clone_model
from keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed, Masking, Concatenate, Flatten, Bidirectional, RepeatVector, Add, Subtract, Lambda
from keras.backend import function as keras_function
from keras.initializers import Constant
from keras.initializers import he_normal
from keras.activations import relu, linear
from keras.optimizers import adam
import keras.backend as K
import keras

from prioritized import PriorityBuffer
from buffer import BaseBuffer

class DRL():
	def __init__(self, envt, load_model_loc=''):
		self.envt = envt
		self.gamma = 0.9
		self.batch_size_fit = 32 # Number of samples per batch to use during fitting
		self.batch_size_predict = 32 # Number of samples per batch to use during prediction
		self.target_update_tau = 0.1
		self.num_min_train_samples = 1000 # Minimum size of replay buffer needed to begin sampling
		self.num_samples = 50
		self.input_size = ((self.envt.num_agents) * 4) + 2 + 5 + 30
		self.output_size = 2
		self.learning_rate = 0.00025
		
		self.epsilon = 1
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.99987
		self.losses = []

		losses = []

		self.training_iterations = 0

		# Get Replay Buffer
		min_len_replay_buffer = 1e6 / self.envt.num_agents # What is the size of the replay buffer???
		epochs_in_episode = (self.envt.stop_epoch - self.envt.start_epoch) / self.envt.epoch_length
		len_replay_buffer = max((min_len_replay_buffer, epochs_in_episode))
		self.replay_buffer = PriorityBuffer(int(len_replay_buffer), 0.6)

		# Get NN Model
		self.model = load_model(load_model_loc) if load_model_loc else self._init_NN()

		# Get target-NN
		self.target_model = clone_model(self.model)
		self.target_model.set_weights(self.model.get_weights())

	def _init_NN(self):
		model = Sequential()
		init_values = he_normal(seed=None)
		model.add(Dense(32, input_dim=self.input_size, activation=relu, kernel_initializer=init_values))
		model.add(Dense(64, activation=relu, kernel_initializer=init_values))
		model.add(Dense(64, activation=relu, kernel_initializer=init_values))
		model.add(Dense(32, activation=relu, kernel_initializer=init_values))
		model.add(Dense(self.output_size, activation=linear))
		opt = adam(lr=self.learning_rate, clipnorm=1.0)
		model.compile(loss='mse', optimizer=opt)
		return model

	# def _init_NN(self):
	# 	init_values = he_normal(seed=None)
	# 	x_input = Input(shape=(self.input_size,), name='State')
	# 	x = Dense(32, activation=relu, kernel_initializer=init_values)(x_input)
	# 	x = Dense(64, activation=relu, kernel_initializer=init_values)(x)
	# 	x = Dense(64, activation=relu, kernel_initializer=init_values)(x)
	# 	x = Dense(32, activation=relu, kernel_initializer=init_values)(x)

	# 	head_v = Dense(16, activation='relu', kernel_initializer=init_values)(x)
	# 	head_v = Dense(1, activation='linear', name="Value")(head_v)
	# 	head_v = RepeatVector(self.output_size)(head_v)
	# 	head_v = Flatten()(head_v)

	# 	head_a = Dense(16, activation='relu', kernel_initializer=init_values)(x)
	# 	head_a = Dense(self.output_size, activation='linear', name='Activation')(head_a)

	# 	m_head_a = RepeatVector(self.output_size)(Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(head_a))
	# 	m_head_a = Flatten(name='meanActivation')(m_head_a)


	# 	corrected_head_a = Subtract()([head_a, m_head_a])

	# 	head_q = Add(name="Q-value")([head_v, corrected_head_a])
	# 	model = keras.Model(inputs=[x_input], outputs=[head_q])
	# 	opt = adam(lr=self.learning_rate, clipnorm=1.0)
	# 	model.compile(loss='mse', optimizer=opt)
	# 	return model

	def _add_locations(self, locations):
		locations_input = np.zeros(shape=(self.envt.car_capacity,), dtype='int32')
		for loc_id in range(len(locations)):
			locations_input[loc_id] = locations[loc_id].destination + 1
		return locations_input

	def _add_delays(self, orders, current_time):
		delays = np.zeros(shape=(self.envt.car_capacity,1)) - 1
		for i, order in enumerate(orders):
			delays[i] = (order.deadline - current_time) / (order.deadline - order.origin_time)
		return delays

	def remember(self, experience):
		self.replay_buffer.add(experience)

	def huber_loss(self, loss):
		return 0.5 * loss ** 2 if abs(loss) < 1.0 else abs(loss) - 0.5

	def update(self, central_agent):
		# Check if replay buffer has enough samples for an update
		if (self.num_min_train_samples > self.replay_buffer.size):
			return

		beta = min(1, 0.4 + 0.6 * (self.envt.num_days_trained / 200.0))
		mini_batch, indices, weights = self.replay_buffer.sample(self.num_samples, beta)

		next_q_val = self.model.predict_on_batch(mini_batch.next_state)
		actions = np.argmax(next_q_val, axis=1)
		next_q_values = self.target_model.predict_on_batch(mini_batch.next_state)
		ind = np.array([i for i in range(self.num_samples)])
		q_values = self.model.predict_on_batch(mini_batch.state)
		target_q_values = q_values.copy()
		### Bellman Update ###
		td_target = mini_batch.reward.squeeze() + self.gamma * next_q_values[[ind], [actions]] * (1 - mini_batch.terminal.squeeze())
		target_q_values[[ind], [mini_batch.action.squeeze()]] = td_target
		error = [self.huber_loss(target_q_values[i, actions[i]] - q_values[i, actions[i]]) for i in range(mini_batch.state.shape[0])]
		self.replay_buffer.update_priority(indices, error)
		result = self.model.fit(mini_batch.state, target_q_values, sample_weight=weights, epochs=1, verbose=0)
		self.losses.append(result.history['loss'][0])

	def get_action(self, state, is_training):
		actions = self.model.predict(state)
		if is_training:
			if np.random.rand() < self.epsilon:
				return np.random.randint(self.output_size)
			else:
				return [max(enumerate(act), key=lambda x: x[1]) for act in actions][0][0]
		else:
			return [max(enumerate(act), key=lambda x: x[1]) for act in actions][0][0]

	def hard_update(self):
		q_network_theta = self.model.get_weights()
		self.target_model.set_weights(q_network_theta)