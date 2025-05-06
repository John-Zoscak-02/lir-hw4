import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from tqdm import tqdm
import matplotlib.pyplot as plt

# Check if CUDA is available
# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')

m = 1
l=1
b=0.1
g=9.8
gamma=0.99

device = torch.device('cpu')

class u_t(nn.Module):
    def __init__(s, xdim=2, udim=1):
        super().__init__()
        """
        Build two layer neural network
        We will assume that the variance of the stochastic
        controller is a constant, so the network simply
        outputs the mean. We will do a hack to code up the constraint
        on the magnitude of the control input. We use a tanh nonlinearity
        to ensure that the output of the network is always between [-1,1]
        and then add a noise to it. While the final sampled control may be larger
        than the constraint we desire [-1,1], this is a quick cheap way to enforce the constraint.
        """
        s.m = nn.Sequential(
                nn.Linear(xdim, 8),
                nn.ReLU(True),
                nn.Linear(8, udim),
                nn.Tanh(),
                )
        s.std = 1

    def forward(s, x, u=None):
        """
        This is a PyTorch function that runs the network
        on a state x to output a control u. We will also output
        the log probability log u_theta(u | x) because we want to take
        the gradient of this quantity with respect to the parameters
        """
        xyw = torch.stack((torch.sin(x[:, 0]), -torch.cos(x[:, 0]), x[:, 1]), 1)
        # mean control
        mu = s.m(xyw)
        # Build u_theta(cdot | x)
        n = Normal(mu, s.std)
        # sample a u if we are simulating the system, use the argument
        # we are calculating the policy gradient
        if u is None:
            u = n.rsample()
        logp = n.log_prob(u)
        return u, logp
    
def get_rev(z, zdot, u): 
    return -0.5*((np.pi-z)**2 + zdot**2 + 0.01*u**2)

def test_policy(policy, z, zdot): 
    return policy(torch.tensor([[z, zdot]]))

def rollout(policy):
    """
    We will use the control u_theta(x_t) to take the control action at each
    timestep. You can use simple Euler integration to simulate the ODE forward
    for T = 200 timesteps with discretization dt=0.05.
    At each time-step, you should record the state x,
    control u, and the reward r
    """
    xs = [np.zeros(2)]; us = []; rs= [];
    dt = 0.05
    # we will compute a policy for a 10 second trajectory, with a 0.05 time-step and T=200 time-steps
    for t in np.arange(0, 10, dt):
        # The interface between PyTorch and numpy becomes a bit funny
        # but all that this line is doing is that it is running u(x) to get
        # a control for one state x
        # @ x0, this is the initial state and the control policy will be an initial control policy. 
        u = policy(torch.from_numpy(xs[-1]).view(1,-1).float())[0].detach().numpy().squeeze().item()

        z, zdot = xs[-1][0], xs[-1][1]
        zp = z + zdot*dt
        zdotp = zdot + dt*(u - m*g*l*np.sin(z) - b*zdot)/m/l**2

        rs.append(get_rev(z, zdot, u))
        us.append(u)
        xs.append(np.array([zp, zdotp]))

    # For a single trajectory, this is an appropriate way to compute the sum of discounted rewards
    # For a multi-trajectory, input space, this will only compute the sum of discounted rewards for a single trajectory. 
    R = sum([rr*gamma**k for k,rr in enumerate(rs)])
    return {'x': torch.tensor(xs[:-1]).float(),
            'u': torch.tensor(us).float(),
            'r': torch.tensor(rs).float(), 'R': R}

# def rollout(policy):
#     """
#     We will use the control u_theta(x_t) to take the control action at each
#     timestep. You can use simple Euler integration to simulate the ODE forward
#     for T = 200 timesteps with discretization dt=0.05.
#     At each time-step, you should record the state x,
#     control u, and the reward r
#     """
#     horizon = 10
#     dt = 0.05
# 
#     T = math.floor(horizon / dt)
#     xs = torch.zeros(2, dtype=torch.float32)[None, None, ...].repeat((T+1, 1, 1)).to(device)
#     us = torch.zeros((T, 1), dtype=torch.float32, device=device)
#     rs = torch.zeros((T, 1), dtype=torch.float32, device=device)
#     # we will compute a policy for a <horizon> second trajectory, with a 0.05 time-step and T=200 time-steps
#     for i, t in enumerate(np.arange(0, horizon, dt)):
#         # x0, this is the initial state and the control policy will be an initial control policy. 
#         states = xs[i].view(1,-1).to(device)
#         u = policy(states)[0]
# 
#         z, zdot = xs[i][0][0], xs[i][0][1]
#         zp = z + zdot*dt
#         zdotp = zdot + dt*(u - m*g*l*torch.sin(z) - b*zdot)/m/l**2
# 
#         rs[i] = get_rev(z, zdot, u)
#         us[i] = u
#         xs[i+1, 0, 0] = zp
#         xs[i+1, 0, 1] = zdotp
# 
#     # For a single trajectory, this is an appropriate way to compute the sum of discounted rewards
#     # For a multi-trajectory, input space, this will only compute the sum of discounted rewards for a single trajectory. 
#     R = sum([rr*gamma**k for k, rr in enumerate(rs)])
#     return {'x': xs[:-1].requires_grad_(),
#             'u': us.requires_grad_(),
#             'r': rs, 
#             'R': R}

def example_train():
    """
    The following code shows how to compute the policy gradient and update
    the weights of the neural network using one trajectory.
    """
    policy = u_t(xdim=2, udim=1)
    optim = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # 1. get a trajectory
    t = rollout(policy)
    """"
    2. We now want to calculate grad log u_theta(u | x), so
    we will feed all the states from the trajectory again into the network
    and this time we are interested in the log-probabilities. The following
    code shows how to update the weights of the model using one trajectory
    """
    logp = policy(t['x'].view(-1,2), t['u'].view(-1,1))[1]
    f = -(t['R']*logp).mean()

    # zero_grad is a PyTorch peculiarity that clears the backpropagation
    # gradient buffer before calling the next .backward()
    policy.zero_grad()
    # .backward computes the gradient of the policy gradient objective with respect
    # to the parameters of the policy and stores it in the gradient buffer
    f.backward()
    # .step() updates the weights of the policy using the computed gradient
    optim.step()


def train():
    """
    Sample multiple trajectory at each iteration and run the training for about 1000
    iterations. You should track the average value of the return across multiple
    trajectories and plot it as a function of the number of iterations.
    """
    NUM_TRAJECTORIES = [2, 4, 8, 12, 16, 32]
    NUM_ITERATIONS = 1000
    RETURN_MIN = 100 
    xdim = 3
    udim = 1

    best_return = -100000000
    best_returns = []
    best_policy = None
    best_n_trajectories = 10000
    for N_TRAJECTORIES in NUM_TRAJECTORIES:
        policy = u_t(xdim, udim)
        optim = torch.optim.Adam(policy.parameters(), lr=1e-3)
        policy.to(device)
        # optim.to(device)

        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(2,1)
        ax[0].set_ylabel("Average Return")
        ax[0].set_title("Average Return vs. Iteration")
        ax[0].grid(True)
        ax[1].set_xlabel("Iteration")
        ax[1].set_ylabel("Average weighted f_means")
        ax[1].set_title("Average weighted f_means loss vs. Iteration")
        ax[1].grid(True)
        line1, = ax[0].plot([], [], 'b-')  # Initialize empty plot line
        line2, = ax[1].plot([], [], 'g-')  # Initialize empty plot line

        iterations = []
        avg_returns = []
        iteration = 0
        # losses = []
        f_means = []
        for iteration in (piterbar := tqdm(range(NUM_ITERATIONS))):
            ts = []
            logps = []
            fs = []
            # loss = 0
            for i in (ptrajbar := tqdm(range(N_TRAJECTORIES), leave=False)):
            # for i in range(NUM_TRAJECTORIES):
                """
                1. get multiple trjaectories
                """
                t = rollout(policy)
                ts.append(t)

                """"
                2. We now want to calculate grad log u_theta(u | x), so we will feed all 
                the states from the trajectory again into the network and this time we are 
                interested in the log-probabilities.
                """
                states = ts[-1]['x'].view(-1,2)# .to(device)
                actions = ts[-1]['u'].view(-1,1)# .to(device)
                # returns = ts[-1]['r'].to(device)
                Rs = ts[-1]['R']

                logp = policy(states, actions)[1]
                # Compute the policy gradient objective for updating the weights
                # grad_logp = torch.autograd.grad(logp.sum(), policy.parameters(), retain_graph=True)
                # policy_grads.append(grad_logp)
                logps.append(logp)

                f = -(Rs*logp).mean()
                fs.append(f)

                # loss += f

                # print(f"Trajectory {i}, f: {f} average logp loss, R: {Rs}")
                ptrajbar.set_description(f"Trajectory {i}, f: {f} average logp loss, R: {Rs}")

            # Compute the average of the logp policy gradient losses across all trajectories
            f_mean = torch.tensor(fs).mean()
            f_means.append(f_mean)
            # loss /= NUM_TRAJECTORIES
            # losses.append(loss.item())

            policy.zero_grad()
            # Compute the gradient of the policy gradient objective with respect to the parameters
            # of the policy and store it in the gradient buffer
            f_mean.backward()
            optim.step()

            # Track average return
            returns = [t['R'].item() for t in ts]
            avg_return = np.mean(returns)
            avg_returns.append(avg_return)

            iterations.append(iteration)
            # Update plot for Average Return (line1)
            line1.set_xdata(iterations)
            line1.set_ydata(avg_returns)
            # Update plot for Train Loss (line2)
            line2.set_xdata(iterations)
            line2.set_ydata(f_means)

            # Adjust plot limits dynamically
            ax[0].relim()
            ax[0].autoscale_view()
            ax[1].relim()
            ax[1].autoscale_view()

            # Redraw plot
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)  # Small pause to allow plot update

            # Print progress
            # print(f"Iteration {iteration}/{NUM_ITERATIONS}, Avg Return: {avg_return:.2f}, Loss: {loss:.2f}")
            piterbar.set_description(f"Iteration {iteration}/{NUM_ITERATIONS}, Avg Return: {avg_return:.2f}")

            if avg_return > RETURN_MIN:
                print(f"Finished training, return={avg_return.item()}")
                break;

        if best_return > avg_returns[-1]:
            best_return = avg_returns[-1]
            best_returns = avg_returns
            best_n_trajectories = N_TRAJECTORIES
            best_policy = policy

    return best_policy, best_returns

def main():
    """
    This is the main function that runs the training loop
    """
    policy, returns = train()
    # example_train()

main()