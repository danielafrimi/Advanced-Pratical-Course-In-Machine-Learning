import torch
import torch.nn.functional as F
import torch.optim as optim
from memory import ReplayMemory, Transition
import random
from torch.utils.tensorboard import SummaryWriter
from base_policy import BasePolicy
import gym
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class QPolicy(BasePolicy):
    # TODO partial code for q-learning
    # you should complete it. You can change it however you want.
    def __init__(self, buffer_size, gamma, model, action_space: gym.Space, summery_writer: SummaryWriter, lr):
        super(QPolicy, self).__init__(buffer_size, gamma, model, action_space, summery_writer, lr)

        # How much you accept the new value vs the old value
        self.lr = lr
        self.optimizer = optim.RMSprop(self.model.parameters())

    def select_action(self, state, epsilon, global_step=None):
        """
        select an action to play
        :param state: 1x9x9x10 (nhwc), n=1, h=w=9, c=10 (types of thing on the board, one hot encoding)
        we define each option(dollar, block, num_player..) as one-hot vector
        each state is a 9x9 area around your snakes head, represented as one-hot vector.
        :param epsilon: maintain balance exploration/exploitation (choose random step or known one)
        :param global_step: used for tensorboard logging
        :return: return single action as integer (0, 1 or 2).
        """

        random_number = random.random()
        if random_number > epsilon:
            # Exploit: select the action with max value (future reward)
            # TODO
            with torch.no_grad():
                expected_reward_per_action = self.model(state)
                # return the action with the maximal expected reward for this state
                return int(np.argmax(expected_reward_per_action))
        else:
            # Explore: select a random action
            return self.action_space.sample()

    def optimize(self, batch_size, global_step=None):
        """
        Optimize the model
        :param batch_size:
        :param global_step:
        :return:
        """


        if len(self.memory) < batch_size:
            return None

        self.memory.batch_size = batch_size
        for transitions_batch in self.memory:

            # transform list of tuples into a tuple of lists.
            # explanation here: https://stackoverflow.com/a/19343/3343043
            batch = Transition(*zip(*transitions_batch))

            # TODO CHANGE  taken from pytorch?
            #  Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

            state_batch = torch.cat(batch.state)
            # TODO delete?
            next_state_batch = torch.cat(batch.next_state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken.
            # These are the actions which would've been taken for each batch state according to policy_net
            # Squeezing index to fit net output dimension
            state_action_values = self.model(state_batch).gather(dim=1, index=action_batch.unsqueeze(-1))

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(batch_size)
            # TODO needs to use targernet? or use the same one?
            next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch

            # Compute Huber loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()

            print('loss is {}'.format(loss))

            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()


