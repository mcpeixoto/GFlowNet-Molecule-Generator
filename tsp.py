import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import random
import os
import time

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Load TSP benchmark data (e.g., from TSPLIB)
def load_tsp_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    coords = []
    node_section = False
    for line in lines:
        if 'NODE_COORD_SECTION' in line:
            node_section = True
            continue
        if 'EOF' in line or 'DISPLAY_DATA_SECTION' in line:
            break
        if node_section:
            parts = line.strip().split()
            coords.append([float(parts[1]), float(parts[2])])
    return np.array(coords)

# Example: Generate synthetic data if no file is provided
def generate_synthetic_tsp(num_cities):
    coords = np.random.rand(num_cities, 2) * 100
    return coords

# Define the environment for TSP
class TSPEnvironment:
    def __init__(self, coords):
        self.coords = coords
        self.num_cities = len(coords)
        self.distance_matrix = distance_matrix(coords, coords)
        self.reset()
    
    def reset(self):
        self.visited = [False] * self.num_cities
        self.current_city = random.randint(0, self.num_cities - 1)
        self.visited[self.current_city] = True
        self.tour = [self.current_city]
        return self.get_state()
    
    def get_state(self):
        return {
            'current_city': self.current_city,
            'visited': self.visited.copy(),
            'tour': self.tour.copy()
        }
    
    def get_available_actions(self):
        return [i for i, visited in enumerate(self.visited) if not visited]
    
    def step(self, action):
        self.current_city = action
        self.visited[action] = True
        self.tour.append(action)
        done = all(self.visited)
        return self.get_state(), done
    
    def compute_total_distance(self, tour=None):
        if tour is None:
            tour = self.tour
        tour = tour + [tour[0]]  # Return to start
        total_distance = sum(
            self.distance_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1)
        )
        return total_distance
    
    def compute_reward(self):
        total_distance = self.compute_total_distance()
        # Reward inversely proportional to total distance
        reward = 1.0 / (total_distance + 1e-6)
        return reward

# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, num_cities):
        super(PolicyNetwork, self).__init__()
        self.num_cities = num_cities
        self.fc = nn.Sequential(
            nn.Linear(num_cities * 2 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, num_cities),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state_vector):
        return self.fc(state_vector)

# Convert state to feature vector
def state_to_feature_vector(state, num_cities):
    current_city_one_hot = np.zeros(num_cities)
    current_city_one_hot[state['current_city']] = 1.0
    visited = np.array(state['visited'], dtype=float)
    tour_length = len(state['tour']) / num_cities  # Normalize
    feature_vector = np.concatenate((current_city_one_hot, visited, [tour_length]))
    return torch.tensor(feature_vector, dtype=torch.float)

# Training parameters
num_iterations = 5000
gamma = 0.95  # Discount factor

def train_gflownet(env, policy_net, optimizer):
    loss_history = []
    reward_history = []
    start_time = time.time()

    for iteration in range(num_iterations):
        state = env.reset()
        trajectory = []
        done = False

        while not done:
            state_vector = state_to_feature_vector(state, env.num_cities)
            action_probs = policy_net(state_vector)
            available_actions = env.get_available_actions()
            action_mask = torch.zeros(env.num_cities)
            action_mask[available_actions] = 1.0
            masked_action_probs = action_probs * action_mask
            if masked_action_probs.sum().item() == 0:
                masked_action_probs = action_mask
            masked_action_probs = masked_action_probs / masked_action_probs.sum()
            action_dist = torch.distributions.Categorical(masked_action_probs)
            action = action_dist.sample().item()

            trajectory.append((state_vector, action))

            state, done = env.step(action)

        # Compute reward
        reward = env.compute_reward()

        # Backpropagation
        loss = 0.0
        R = reward
        for state_vector, action in reversed(trajectory):
            action_probs = policy_net(state_vector)
            log_prob = torch.log(action_probs[action] + 1e-10)
            loss += -log_prob * R
            R *= gamma

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        reward_history.append(reward)

        if (iteration + 1) % 100 == 0:
            avg_reward = np.mean(reward_history[-100:])
            print(f"Iteration {iteration + 1}, Loss: {loss.item():.4f}, Avg Reward: {avg_reward:.6f}")

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    return loss_history, reward_history, training_time

# Visualize the tour
def plot_tour(coords, tour, title="TSP Tour", filename=None):
    tour_coords = coords[tour + [tour[0]]]  # Return to start
    plt.figure(figsize=(8, 6))
    plt.plot(tour_coords[:, 0], tour_coords[:, 1], 'o-')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    if filename:
        plt.savefig(filename)
    plt.show()

# Main function
def main():
    # Load TSP data
    # For benchmarking, use a TSPLIB instance with a known optimal tour length
    # For example, 'berlin52.tsp' with optimal tour length 7542
    tsp_file = 'berlin52.tsp'
    coords = load_tsp_data(tsp_file)
    num_cities = len(coords)
    optimal_tour_length = 7542  # Known optimal tour length for berlin52

    # Initialize environment and policy network
    env = TSPEnvironment(coords)
    policy_net = PolicyNetwork(num_cities)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

    # Train GFlowNet
    loss_history, reward_history, training_time = train_gflownet(env, policy_net, optimizer)

    # Plot training statistics
    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label="Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()

    # Reward Plot
    plt.subplot(1, 2, 2)
    plt.plot(reward_history, label="Reward", color="orange")
    plt.xlabel("Iterations")
    plt.ylabel("Reward")
    plt.title("Reward Over Time")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_statistics.png")
    plt.show()

    # Sample a tour from the trained policy
    start_time = time.time()
    state = env.reset()
    tour = [state['current_city']]
    done = False

    while not done:
        state_vector = state_to_feature_vector(state, env.num_cities)
        with torch.no_grad():
            action_probs = policy_net(state_vector)
        available_actions = env.get_available_actions()
        action_mask = torch.zeros(env.num_cities)
        action_mask[available_actions] = 1.0
        masked_action_probs = action_probs * action_mask
        if masked_action_probs.sum().item() == 0:
            masked_action_probs = action_mask
        masked_action_probs = masked_action_probs / masked_action_probs.sum()
        action_dist = torch.distributions.Categorical(masked_action_probs)
        action = action_dist.sample().item()

        tour.append(action)
        state, done = env.step(action)

    solution_time = time.time() - start_time

    # Compute tour length
    found_tour_length = env.compute_total_distance(tour)
    optimality_gap = (found_tour_length - optimal_tour_length) / optimal_tour_length * 100

    print(f"Found Tour Length: {found_tour_length:.2f}")
    print(f"Optimal Tour Length: {optimal_tour_length:.2f}")
    print(f"Optimality Gap: {optimality_gap:.2f}%")
    print(f"Solution Generation Time: {solution_time:.4f} seconds")

    # Plot the tour
    plot_tour(coords, tour, title="Sampled TSP Tour", filename="sampled_tour.png")

if __name__ == "__main__":
    main()
