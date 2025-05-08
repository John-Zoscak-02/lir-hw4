import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import random 
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')
plt.rcParams['figure.raise_window'] = False

def rollout(e, q, eps=0, T=200):
    traj = []

    x = e.reset()[0]
    for t in range(T):
        u = q.control(torch.from_numpy(x).float().unsqueeze(0), eps=eps, action_space=e.action_space)
        u = u.int().numpy().squeeze()

        xp,r,terminated, truncated, info = e.step(u)
        t = dict(x=x,
                 xp=xp,
                 r=r,
                 u=u,
                 terminated=terminated,
                 truncated=truncated,
                 info=info)
        x = xp
        traj.append(t)
        if terminated or truncated:
            break
    return traj

class q_t(nn.Module):
    def __init__(s, xdim, udim, hdim=16):
        super().__init__()
        s.xdim, s.udim = xdim, udim
        s.m = nn.Sequential(
                            nn.Linear(xdim, hdim),
                            nn.ReLU(True),
                            nn.Linear(hdim, udim),
                            )
    def forward(s, x):
        return s.m(x)

    def control(s, x, eps=0, action_space=spaces.Discrete(2)):
        # 1. get q values for all controls
        q = s.m(x)

        # eps-greedy strategy to choose control input
        # note that for eps=0
        # you should return the correct control u
        eps_sample = random.random()
        u = 0
        if (eps_sample < 1 - eps):
            # repeat the controls taken in the past 
            # because the udim probably equals 2, we can just assume the q value of the zeroth index is the liklihood of 
            # moving left and that the first index is the liklihood of moving right 
            u = torch.argmax(q)
        else:
            # sample uniformly from the entire controls space 
            u = torch.tensor(action_space.sample())
        return u

def loss(q1, q2, ds, target_q1, target_q2, gamma=0.995, alpha=0.05):
    # 1. sample mini-batch from datset ds
    # mini-batch should be picked to have samples from different trajectories
    random_t = np.random.choice(ds, size=10)
    mini_batch = [d[np.random.randint(low=0, high=len(d))] for d in random_t]

    # 2. code up dqn with double-q trick
    fs1a = 0  
    fs2a = 0 
    for sample in mini_batch:
        state = sample['x']
        action = sample['u'] 
        reward = sample['r']
        next_state = sample['xp'] 
        terminated = sample['terminated'] or sample['truncated']
        terminated = torch.tensor(int(terminated), dtype=torch.float32).unsqueeze(0)

        qp1 = q1.forward(torch.from_numpy(next_state))
        up1 = torch.argmax(qp1, keepdim=True) 
        tp2 = target_q2.forward(torch.from_numpy(next_state))[up1]
        target_ddqn1_value = ((reward) + 
                              (gamma * (1 - int(terminated))) * tp2)

        qp2 = q2.forward(torch.from_numpy(next_state))
        up2 = torch.argmax(qp2, keepdim=True) 
        tp1 = target_q1.forward(torch.from_numpy(next_state))[up2]
        target_ddqn2_value = ((reward) + 
                              (gamma * (1 - int(terminated))) * tp1)

        f1 = ((q1(torch.from_numpy(state))) - 
              # (reward) - 
              # (gamma*(1 - int(terminated))*target_ddqn2_value))**2
              target_ddqn1_value.detach()) **2 
        f2 = ((q2(torch.from_numpy(state))) - 
              # (reward) - 
              # (gamma*(1 - int(terminated))*target_ddqn1_value))**2
              target_ddqn2_value.detach()) **2 
            
        fs1a += f1.mean()
        fs2a += f2.mean()

    # 3. return the objectives f
    return fs1a/len(mini_batch), fs2a/len(mini_batch)

# u* = argmax q(x', u)
# (q(x, u) - r - g*(1-indicator of terminal)*qc(x' , u*))**2

def evaluate(q, e=gym.make('CartPole-v0'), gamma=0.995):
    # 1. create a new environment e (Just use the default param....)
    # e = gym.make('CartPole-v0')

    # 2. run the learnt q network for 100 trajectories on
    # this new environment to take control actions. Remember that
    # you should not perform epsilon-greedy exploration in the evaluation
    # phase
    # and report the average discounted
    # return of these 100 trajectories
    q.eval()
    Rs = []
    for i in range(100):
        t = rollout(e, q, eps=0, T=200)
        rs = [ti['r'] for ti in t]
        R = sum([rr*gamma**k for k,rr in enumerate(rs)])
        Rs.append(R)
    q.train()
    e.render()
    return Rs

if __name__=='__main__':
    e = gym.make('CartPole-v0')

    xdim, udim =    e.observation_space.shape[0], \
                    e.action_space.n
    alpha = 0.05
    gamma = 0.995

    q1 = q_t(xdim, udim, 8)
    q2 = q_t(xdim, udim, 8)
    target_q1 = q_t(xdim, udim, 8)
    target_q2 = q_t(xdim, udim, 8)
    # Initialize target networks with the same weights
    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())
    # Adam is a variant of SGD and essentially works in the
    # same way
    rho1 = q1.parameters()
    rh02 = q2.parameters()
    optim1 = torch.optim.Adam(rho1, lr=1e-3,
                          weight_decay=1e-4)
    optim2 = torch.optim.Adam(rh02, lr=1e-3,
                          weight_decay=1e-4)

    ds = []

    # collect few random trajectories with
    # eps=1
    for i in range(10000):
        ds.append(rollout(e, q1, eps=1, T=200))

    # for i in range(10000):
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(3,1, figsize=(16, 10))
    ax[0].set_ylabel("Average Return")
    ax[0].set_title("Average Return vs. Iteration")
    ax[0].grid(True)
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Average weighted f_means")
    ax[1].set_title("Average weighted f_means vs. Iteration")
    ax[1].grid(True)
    ax[2].set_xlabel("Iteration")
    ax[2].set_ylabel("Average weighted f_means")
    ax[2].set_title("Average weighted f_means vs. Iteration")
    ax[2].grid(True)
    line1, = ax[0].plot([], [], 'b-')  # Initialize empty plot line
    line2, = ax[1].plot([], [], 'g-')  # Initialize empty plot line
    line3, = ax[2].plot([], [], 'r-')  # Initialize empty plot line

    iterations = []
    f1s = []
    f2s = []
    Rs = []
    for i in tqdm(range(10000)):
        q1.train()
        q2.train()
        t = rollout(e, q1 if i % 2 == 0 else q2)
        ds.append(t)

        # perform weights updates on the q network
        # need to call zero grad on q function
        # to clear the gradient buffer
        optim1.zero_grad()
        optim2.zero_grad()
        f1, f2 = loss(q1, q2, ds, target_q1, target_q2)
        f1s.append(f1.mean().detach())
        f2s.append(f2.mean().detach())
        f1.backward()
        f2.backward()
        optim1.step()
        optim2.step()

        if (i+1)%1000 == 0:
            # print("EVALUATING ON THE TRAINING ENVIRONMENT")
            Rs1 = evaluate(q1 if i % 2 == 0 else q2, e)
            # print("EVALUATING ON A NEW ENVIRONMENT")
            Rs2 = evaluate(q1 if i % 2 == 0 else q2)

            print("DISCOUNTED REWARD IN THE TRAINING ENVIRONMENT = ", sum(Rs1) / len(Rs1))
            plt.figure()
            fig, ax = plt.subplots(2,1, figsize=(16, 10))
            ax[0].plot(Rs1)
            ax[0].set_ylabel("Average Return")
            ax[0].set_title("Training Environment: Average Return vs. Iteration")
            ax[0].grid(True)
            print("DISCOUNTED REWARD ON A NEW ENVIRONMENT = ", sum(Rs2) / len(Rs2))
            ax[1].plot(Rs2)
            ax[1].set_ylabel("Average Return")
            ax[1].set_title("New Environment: Average Return vs. Iteration")
            ax[1].grid(True)
            plt.show(block=False)

            plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots(3,1, figsize=(16, 10))
            ax[0].set_ylabel("Average Return")
            ax[0].set_title("Average Return vs. Iteration")
            ax[0].grid(True)
            ax[1].set_xlabel("Iteration")
            ax[1].set_ylabel("Average weighted f_means")
            ax[1].set_title("Average weighted f_means vs. Iteration")
            ax[1].grid(True)
            ax[2].set_xlabel("Iteration")
            ax[2].set_ylabel("Average weighted f_means")
            ax[2].set_title("Average weighted f_means vs. Iteration")
            ax[2].grid(True)
            line1, = ax[0].plot([], [], 'b-')  # Initialize empty plot line
            line2, = ax[1].plot([], [], 'g-')  # Initialize empty plot line
            line3, = ax[2].plot([], [], 'r-')  # Initialize empty plot line

        
        rs = [ti['r'] for ti in t]
        R = sum([rr*gamma**k for k,rr in enumerate(rs)])
        Rs.append(R)

        # f_means_copy = [f_mean.detach() for f_mean in f_means]
        iterations.append(i)
        # Update plot for Average Return (line1)
        line1.set_xdata(iterations)
        line1.set_ydata(Rs)
        # Update plot for Train fmeans (line2)
        line2.set_xdata(iterations)
        line2.set_ydata(f1s)
        # Update plot for Train fmeans (line2)
        line3.set_xdata(iterations)
        line3.set_ydata(f2s)

        # Adjust plot limits dynamically
        ax[0].relim()
        ax[0].autoscale_view()
        ax[1].relim()
        ax[1].autoscale_view()
        ax[2].relim()
        ax[2].autoscale_view()

        # Redraw plot
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)  # Small pause to allow plot update


        # Exponential averaging for target networks
        with torch.no_grad():
            for param, target_param in zip(q1.parameters(), target_q1.parameters()):
                target_param.data.copy_((1 - alpha) * target_param.data + alpha * param.data)
            for param, target_param in zip(q2.parameters(), target_q2.parameters()):
                target_param.data.copy_((1 - alpha) * target_param.data + alpha * param.data)
                