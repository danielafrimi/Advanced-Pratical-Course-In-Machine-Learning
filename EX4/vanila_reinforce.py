import torch
import torch.nn.functional as F
from memory import ReplayMemory, Transition
from torch.utils.tensorboard import SummaryWriter
from base_policy import BasePolicy
import gym
from torch.distributions import Categorical
import numpy as np




class VanillaPolicy(BasePolicy):
    # partial code for q-learning
    # you should complete it. You can change it however you want.
    def __init__(self, buffer_size, gamma, model, action_space: gym.Space, summery_writer: SummaryWriter, lr):
        super(VanillaPolicy, self).__init__(buffer_size, gamma, model, action_space, summery_writer, lr)

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
        self.model.eval()
        with torch.no_grad():
            # sample from the distribution vector the action
            prob_actions = self.model(state)

            categorical_sample = Categorical(prob_actions)
            action = categorical_sample.sample()
            print("this is the vector_{}".format(prob_actions))
            print("this is action {}".format(action.item()))
            return action.item()


    def optimize(self, T, global_step=None, alpha=0.8):
        # Check if there are enough batches for optimization
        if len(self.memory) < T:
            return None

        self.memory.batch_size = T

        for transitions_batch in self.memory:
            batch = Transition(*zip(*transitions_batch))

            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            next_state_batch = torch.cat(batch.next_state)

            # Calculate Vt where Vt is the discounted sum of all future rewards for the length of the episode.
            V = []
            rewards = reward_batch.numpy()
            T = len(rewards)
            for t in range(T):
                V_t = 0
                for i in range(t, T):
                    V_t = rewards[i] * (self.gamma ** (i - t))
                V.append(V_t)

            # Scale our reward vector by subtracting the mean from each element and scaling to
            # unit variance by dividing by the standard deviation.
            with torch.no_grad():
                V = torch.FloatTensor(V)
                V = (V - V.mean()) / (V.std() + np.finfo(np.float32).eps)
                entropy = Categorical(self.model(state_batch)).entropy()

            pi = (torch.squeeze(self.model(state_batch).gather(dim=1, index=action_batch.unsqueeze(-1))))

            # print("pi equal to {}".format(pi))

            pi = pi.clamp(min=0.2, max=1)
            # print("pi clamp equal to {}".format(pi))

            log_pi = torch.log(pi)
            mul = torch.mul(log_pi, V)
            ent = np.power(alpha, global_step) * entropy
            print("this is mul {}".format(mul))
            print("this is entr {}".format(ent))

            objective_t = torch.mul(log_pi, V) + alpha * entropy
            # print("log pi ".format(log_pi))


            # In order to maximize multiply by -1
            loss = torch.mean(objective_t).mul(-1)
            self.writer.add_scalar('training/objective', -1*loss.item(), global_step)

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm(self.model.parameters(), 2)

            self.optimizer.step()

