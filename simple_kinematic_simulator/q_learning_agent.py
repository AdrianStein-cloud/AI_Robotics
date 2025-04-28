import random
import pickle
import numpy as np

class QLearningAgent:
    def __init__(self, actions, state_bins, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        actions: list of action names, e.g. ['left','right','forward']
        state_bins: list of ints, number of buckets per sensor, e.g. [5,5,5]
        """
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.state_bins = state_bins
        self.q_table = {}  # maps state‐tuple to numpy array of Qs

    def discretize(self, sensor_readings, max_distance):
        """Convert raw distances into a discrete state tuple."""
        state = []
        for i, dist in enumerate(sensor_readings):
            bins = self.state_bins[i]
            # bucket between 0 and bins-1
            b = int(dist / max_distance * bins)
            b = max(0, min(b, bins-1))
            state.append(b)
        return tuple(state)

    def _get_qs(self, state):
        """Return Q‐vector for state, creating if needed."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return self.q_table[state]

    def choose_action(self, state):
        """ε‐greedy action selection."""
        qs = self._get_qs(state)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        # break ties randomly
        max_q = np.max(qs)
        candidates = [a for a, q in zip(self.actions, qs) if q == max_q]
        return random.choice(candidates)

    def learn(self, state, action, reward, next_state):
        """Perform one Q‐learning update."""
        qs      = self._get_qs(state)
        next_qs = self._get_qs(next_state)
        a_idx   = self.actions.index(action)
        target  = reward + self.gamma * np.max(next_qs)
        qs[a_idx] += self.alpha * (target - qs[a_idx])

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)