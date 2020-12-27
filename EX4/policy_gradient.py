import gym
import numpy as np
import torch
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from base_policy import BasePolicy
from memory import Transition


class PolicyGradient(BasePolicy):
    # partial code for q-learning
    # you should complete it. You can change it however you want.
    def __init__(self, buffer_size, gamma, model, action_space: gym.Space, summery_writer: SummaryWriter, lr):
        super(PolicyGradient, self).__init__(buffer_size, gamma, model, action_space, summery_writer, lr)

        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr)

    def select_action(self, state, epsilon, global_step=None):
        """
        select an action to play
        :param state: 1x9x9x10 (nhwc), n=1, h=w=9, c=10 (types of thing on the board, one hot encoding)
        :param epsilon: epsilon...
        :param global_step: used for tensorboard logging
        :return: return single action as integer (0, 1 or 2).
        """

        # Chooses an action based on our policy probability distribution
        # NO EPSILON in this policy, because it the estimator of the unbiased wont be valid
        with torch.no_grad():
            # Sample action from the distribution action-vector
            prob_actions = self.model(state)

            categorical_sample = Categorical(prob_actions)
            sampled_action = categorical_sample.sample()
            return sampled_action.item()


    def optimize(self, batch_size, global_step=None, update_target=None, alpha=0.4):

        # Optimize only when we stored at least batch_size samples
        if len(self.memory) < batch_size:
            return None

        self.memory.batch_size = batch_size

        for transitions_batch in self.memory:
            batch = Transition(*zip(*transitions_batch))

            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            next_state_batch = torch.cat(batch.next_state)

            # Calculate Vt
            rewards = np.zeros(batch_size)
            for t in range(batch_size):
                relevant_rewards = reward_batch[t:]
                gammas = torch.from_numpy(np.array([np.power(self.gamma, i - t) for i in range(t, batch_size)]))
                rewards[t] = torch.sum(relevant_rewards * gammas)

            # Scale our reward vector by subtracting the mean from each element and scaling to
            # unit variance by dividing by the standard deviation.
            rewards = torch.FloatTensor(rewards)
            rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

            prediction_actions = torch.clamp(self.model(state_batch), 0.01, 1 - 0.01)
            entropy = torch.distributions.Categorical(prediction_actions).entropy()

            # Attach each action to its probability
            log_pi = torch.log(torch.squeeze(torch.gather(prediction_actions, 1, torch.unsqueeze(action_batch, 1))))

            objective_t = torch.mul(log_pi, rewards) + alpha * entropy

            # In order to maximize multiply by -1
            loss = -torch.mean(objective_t)
            self.writer.add_scalar('training/loss', loss.item(), global_step)

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
            self.optimizer.step()

