import torch
import torch.nn.functional as F
import torch.optim as optim
from memory import ReplayMemory, Transition
import random
from torch.utils.tensorboard import SummaryWriter
from base_policy import BasePolicy
from q_policy import QPolicy
import gym
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class DoublePolicy(QPolicy):
    # you should complete it. You can change it however you want.
    def __init__(self, buffer_size, gamma, model, target_model, action_space: gym.Space, summery_writer: SummaryWriter, lr):
        super(DoublePolicy, self).__init__(buffer_size, gamma, model, action_space, summery_writer, lr)

        self.target_model = target_model



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
            with torch.no_grad():
                expected_reward_per_action = self.model(state)

                # return the action with the maximal expected reward for this state
                return np.argmax(expected_reward_per_action).item()
        else:
            # Explore: select a random action
            return self.action_space.sample()

    def optimize(self, batch_size, global_step=None, update_target=False, alpha=None):
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

            state_batch = torch.cat(batch.state)
            next_state_batch = torch.cat(batch.next_state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            state_action_values = torch.squeeze(self.model(state_batch).gather(dim=1, index=action_batch.unsqueeze(-1)))

            self.optimizer.zero_grad()

            with torch.no_grad():
                next_state_values = self.target_model(next_state_batch).max(1)[0].detach()
                # Compute the expected Q values
                expected_state_action_values = (next_state_values * self.gamma) + reward_batch

            loss = self.mse_loss(state_action_values, expected_state_action_values)
            self.writer.add_scalar('training/loss', loss, global_step)

            loss.backward()

            torch.nn.utils.clip_grad_norm(self.model.parameters(), 2)
            self.optimizer.step()

            # Evert 'update target' we pass the weights of the model to target one
            if update_target:
                # self.target_model.weight.copy_(self.model.weight)
                self.target_model.load_state_dict(self.model.state_dict())



