import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import QED
from rdkit.Chem.Draw import MolsToGridImage, MolToImage
from IPython.display import display, clear_output
import os

# Define possible atom types
ATOM_TYPES = ['C', 'N', 'O', 'F']
MAX_LENGTH = 5  # Maximum molecule length

# Define reward function
def compute_reward(mol):
    if mol is None:
        return 0.0
    try:
        reward = QED.qed(mol)
    except:
        reward = 0.0
    return reward

# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.fc(x)

# Convert molecule to feature vector
def mol_to_feature_vector(mol):
    atom_counts = [0] * len(ATOM_TYPES)
    if mol is None or mol.GetNumAtoms() == 0:
        return torch.tensor(atom_counts, dtype=torch.float)
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in ATOM_TYPES:
            idx = ATOM_TYPES.index(symbol)
            atom_counts[idx] += 1
    return torch.tensor(atom_counts, dtype=torch.float)

ACTIONS = ATOM_TYPES + ['Terminate']
ACTION_SIZE = len(ACTIONS)

policy_net = PolicyNetwork(state_size=len(ATOM_TYPES), action_size=ACTION_SIZE)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

# Training parameters
num_iterations = 1000
gamma = 0.99  # Discount factor

# Tracking metrics for visualization
loss_history = []
reward_history = []

# Create output directory for images
output_dir = "training_visuals"
os.makedirs(output_dir, exist_ok=True)

# Training with animation and statistics tracking
def train_gflownet():
    for iteration in range(num_iterations):
        mol = Chem.RWMol()
        state = mol_to_feature_vector(None)
        trajectory = []
        terminated = False
        current_molecule_images = []

        while not terminated and mol.GetNumAtoms() < MAX_LENGTH:
            # Get action probabilities
            state_input = state.unsqueeze(0)
            action_probs = policy_net(state_input)
            action_probs = action_probs.squeeze(0)
            action_dist = torch.distributions.Categorical(action_probs)
            action_idx = action_dist.sample()
            action = ACTIONS[action_idx]

            # Track trajectory
            trajectory.append((state, action_idx))

            # Visualization of the current molecule
            current_mol = mol.GetMol()
            if current_mol and current_mol.GetNumAtoms() > 0:
                img = MolToImage(current_mol, size=(150, 150))
                current_molecule_images.append(img)

            # Apply the chosen action
            if action == 'Terminate':
                terminated = True
            else:
                atom = Chem.Atom(action)
                mol.AddAtom(atom)
                state = mol_to_feature_vector(mol)

        # Calculate reward
        final_mol = mol.GetMol()
        reward = compute_reward(final_mol)

        # Backpropagation
        loss = 0.0
        R = reward
        for state, action_idx in reversed(trajectory):
            state_input = state.unsqueeze(0)
            action_probs = policy_net(state_input)
            action_probs = action_probs.squeeze(0)
            log_prob = torch.log(action_probs[action_idx] + 1e-10)
            loss += -log_prob * R
            R *= gamma

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save metrics
        loss_history.append(loss.item())
        reward_history.append(reward)

        # Animation: Display molecule as it’s built
        if iteration % 50 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item():.4f}, Reward: {reward:.4f}")
            clear_output(wait=True)

# Sampling function for trained network
def sample_molecule():
    mol = Chem.RWMol()
    state = mol_to_feature_vector(None)
    terminated = False
    
    while not terminated and mol.GetNumAtoms() < MAX_LENGTH:
        with torch.no_grad():
            state_input = state.unsqueeze(0)
            action_probs = policy_net(state_input)
            action_probs = action_probs.squeeze(0)
        action_dist = torch.distributions.Categorical(action_probs)
        action_idx = action_dist.sample()
        action = ACTIONS[action_idx]
        
        if action == 'Terminate':
            terminated = True
        else:
            atom = Chem.Atom(action)
            mol.AddAtom(atom)
            state = mol_to_feature_vector(mol)
    
    return mol.GetMol()

# Train the GFlowNet with visualization
train_gflownet()

# Plotting training statistics
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

# Save plots
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "training_statistics.png"))
plt.show()

# Sample and visualize molecules
sampled_molecules = [sample_molecule() for _ in range(20)]
sampled_molecules = [mol for mol in sampled_molecules if mol is not None and mol.GetNumAtoms() > 0]

# Display the sampled molecules in a grid
img = MolsToGridImage(sampled_molecules, molsPerRow=5, subImgSize=(200,200))
img.save(os.path.join(output_dir, "sampled_molecules.png"))
img.show()
