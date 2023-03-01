import numpy as np
from collections import namedtuple
import random


class BaseBuffer:
    """ Base class for 1-step buffers. Numpy queue implementation with
    multiple arrays. Sampling efficient in numpy (thanks to fast indexing)

    Arguments:
        - capacity: Maximum size of the buffer
    """

    Transition = namedtuple("Transition",
                            "state action reward next_state terminal")

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.size = 0
        self.pos = 0

    def push(self, transition: Transition):
        """ Push a transition object (with single elements) to the buffer.
        FIFO implementation using <_cycle>. <_cycle> keeps track of the next
        available index to write. Remember to update <size> attribute as we
        push transitions.
        Arguments:
             - transition: Experience transition (state, action, reward, next_state, terminal)
        """
        if self.size < self.capacity:
            self.buffer.append(transition)
            self.size += 1
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Transition:
        """ Uniformly sample a batch of transitions from the buffer. If
        batchsize is less than the number of valid transitions in the buffer
        return None. The return value must be a Transition object with batch
        of state, actions, .. etc.
            Return: T(states, actions, rewards, next_states, terminals)
        """
        if batch_size > self.size:
            return None
        mini_batch = random.sample(self.buffer, batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        terminals = []
        for sample in mini_batch:
            states.append(sample.state)
            actions.append(sample.action)
            rewards.append(sample.reward)
            next_states.append(sample.next_state)
            terminals.append(sample.terminal)
        return self.Transition(np.array(states), np.array(actions, dtype=np.int64).reshape(-1, 1),
                               np.array(rewards, dtype=np.float32).reshape(-1, 1), np.array(next_states),
                               np.array(terminals, dtype=np.float32).reshape(-1, 1))
