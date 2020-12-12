import numpy as np
from ple.games.waterworld import *
from random import Random
from tensorflow import keras
from keras import layers



class my_waterworld(WaterWorld):

	def _agent_act(self, key):
		self.dx = 0
		self.dy = 0
		if key == 'up':
			self.dy -= self.AGENT_SPEED // 4
		if key == 'down':
			self.dy += self.AGENT_SPEED // 4
		if key == 'left':
			self.dx -= self.AGENT_SPEED // 4
		if key == 'right':
			self.dx += self.AGENT_SPEED // 4

	def agent_step(self, dt, action):
		dt /= 1000.0
		self.screen.fill(self.BG_COLOR)

		self.score += self.rewards["tick"]
		self._agent_act(action)
		self.player.update(self.dx, self.dy, dt)

		hits = pygame.sprite.spritecollide(self.player, self.creeps, True)
		for creep in hits:
			self.creep_counts[creep.TYPE] -= 1
			self.score += creep.reward
			self._add_creep()
		
		if self.creep_counts["GOOD"] == 0:
			self.score += self.rewards["win"]
		self.creeps.update(dt)
		
		self.player.draw(self.screen)
		self.creeps.draw(self.screen)

	def getState(self):
		state = super().getGameState()
		state['score'] = self.getScore()
		
		for i in state['creep_pos']['GOOD']:
			i.append(1)
		for i in state['creep_pos']['BAD']:
			i.append(0)


		creeps_pos = np.array(state['creep_pos']['GOOD']).flatten()
		creeps_pos = np.concatenate([creeps_pos, np.array(state['creep_pos']['BAD']).flatten()])
		
		score = state['score'] * 10	

		for i in state['creep_dist']['GOOD']:
			if i <= 210.0:
				score += -0.033 * i + 6.0

		for i in state['creep_dist']['BAD']:
			if i <= 210.0:
				score -= -0.033 * i + 6.0

		game_state = np.concatenate([np.array([state['player_x'], state['player_y'], state['player_velocity_x'], state['player_velocity_y']]), creeps_pos])

		return game_state, score


class ReplayMemory(object):
	def __init__(self, capacity):
		self.capacity = capacity
		

class Agent(object):
	def __init__(self):
		self.actions = ['left','right','up','down']

	def build_model(self):
		inputs = keras.Input(shape=(34,))
		hidden_1 = layers.Dense(128, activation='relu')(inputs)
		hidden_2 = layers.Dense(128, activation='relu')(hidden_1)
		outputs = layers.Dense(4, activation='linear')(hidden_2)

		model = keras.Model(inputs=inputs, outputs=outputs)

		model.compile(loss=lambda x : x**2, optimizer='adam')
		self.model = model	
	
	def predict_move(self, state):
		q_values = self.model.predict(np.array([state]))
		return [self.actions[np.argmax(q_values)], q_values]

	def rand_act(self):
		r = Random()
		return self.actions[r.randint(0,3)]


if __name__ == '__main__':

	pygame.init()
	game = my_waterworld(width=512, height=512, num_creeps=10)
	game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
	game.clock = pygame.time.Clock()
	game.rng = np.random.RandomState(24)
	
	
	agent = Agent()
	agent.build_model()
	memory_size = 100000
	epochs = 10
	num_steps = 15000
	epsilon = 1
	gamma = 0.9

	for epoch in range(1, epochs + 1):
		steps, num_episodes = 0, 0
		losses, rewards = [], []
		game.init()
		
		while not game.game_over() and steps < num_steps:
			dt = game.clock.tick_busy_loop(30)
			current_state, score = game.getState()

			r = Random()
			if r.uniform(0.0, 1.) < epsilon:
				action = agent.rand_act()
			else:
				action = agent.predict_move(current_state)[0]

			game.agent_step(dt, action)
			next_state, next_score = game.getState()
			reward =  next_score - score

			next_action, q_values = agent.predict_move(next_state)
			target = (1 - gamma) * reward + gamma * np.max(q_values)

			agent.model.train_on_batch(np.array([current_state]), [i if index != agent.actions.index(next_action) else target for index, i in enumerate(q_values[0])])
			
			pygame.display.update()
			steps += 1
			epsilon -= 0.001



