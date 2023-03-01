import numpy as np
from buffer import BaseBuffer
from seg_tree import SumTree
from collections import namedtuple


class PriorityBuffer(BaseBuffer):
    """ Prioritized Replay Buffer that sample transitions with a probability
    that is proportional to their respected priorities.
        Arguments:
            - capacity: Maximum size of the buffer
    """

    Transition = namedtuple("Transition",
                            "state action reward next_state terminal")

    def __init__(self, capacity: int, alpha: float, epsilon=0.1):
        super().__init__(capacity)
        self.sum_tree = SumTree(self.capacity)
        self.epsilon = epsilon
        self.alpha = alpha
        self._cycle = 0
        self.size = 0

        self.priorities = np.zeros((self.capacity,), dtype=np.float32)

        self.max_p = epsilon ** alpha

    def push(self, transition: Transition):
        """ Push a transition object (with single elements) to the buffer.
        Transitions are pushed with the current maximum priority (also push
        priorities to both min and sum tree). Remember to set <_cycle> and
        <size> attributes.
            Arguments:
                - transition: Experience transition (state, action, reward, next_state, terminal)
        """
        if self.size < self.capacity:
            self.buffer.append(transition)
            self.size += 1
        else:
            self.buffer[self._cycle] = transition
            self._cycle = (self._cycle + 1) % self.capacity
        self.sum_tree.push(self.max_p)  # push with max priority

    def sample(self, batch_size: int, beta=None) -> (Transition, np.array, np.array):
        """ Sample a transition based on priorities.
            Arguments:
                - batch_size: Size of the batch
                - beta: Importance sampling weighting annealing
            Return:
                - batch of samples
                - indexes of the sampled transitions (so that corresponding
                priorities can be updated)
                - Importance sampling weights which will multiply with loss
        """
        if batch_size > self.size:
            return None

        samples = []
        leaves = []
        priority_segment = self.sum_tree.tree[0] / batch_size
        for i in range(batch_size):
            # A value is uniformly sample from each range
            left = priority_segment * i
            right = priority_segment * (i + 1)
            sample = np.random.uniform(left, right)
            leaf_idx = self.sum_tree.get(sample)
            samples.append(sample)
            leaves.append(leaf_idx)

        sample_priority = self.sum_tree.tree[list(np.array(leaves) + self.capacity -1)]
        sample_prob = sample_priority ** self.alpha
        sample_prob /= sample_prob.sum()
        weights = (sample_prob * len(self.buffer)) ** (-beta)
        weights /= weights.max()

        states = []
        actions = []
        rewards = []
        next_states = []
        terminals = []

        for leaf_id in leaves:
            experience = self.buffer[leaf_id]
            states.append(experience.state)
            actions.append(experience.action)
            rewards.append(experience.reward)
            next_states.append(experience.next_state)
            terminals.append(experience.terminal)

        transition = self.Transition(np.array(states), np.array(actions, dtype=np.int64).reshape(-1, 1),
                               np.array(rewards, dtype=np.float32).reshape(-1, 1), np.array(next_states),
                               np.array(terminals, dtype=np.float32).reshape(-1, 1))

        return transition, np.array(leaves), np.array(weights)

    def update_priority(self, indexes: np.array, values: np.array):
        """ Update the priority values of given indexes (for both min and sum
        trees). Also update max_p value!
         Arguments:
             - indexes(np.array) : Index of experiences that will update
             - values(np.array) : Update value of experience
        """

        # update both trees: (error + epsilon) ** alpha
        for idx, value in zip(indexes, values):
            self.sum_tree.update(idx, (value + self.epsilon) ** self.alpha)

        self.max_p = min(max(self.sum_tree.tree[self.capacity-1:]), 1)


