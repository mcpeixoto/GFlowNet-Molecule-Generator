import torch
import torch.nn as nn
import torch.optim as optim
import random
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import QED
from rdkit.Chem.Draw import MolsToGridImage

# Define possible atom types
ATOM_TYPES = ['C', 'N', 'O', 'F']

# Maximum molecule length
MAX_LENGTH = 5

def compute_reward(mol):
    if mol is None:
        return 0.0
    try:
        reward = QED.qed(mol)
    except:
        reward = 0.0
    return reward

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

def mol_to_feature_vector(mol):
    # Simple feature vector: counts of each atom type
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

def train_gflownet():
    for iteration in range(num_iterations):
        mol = Chem.RWMol()
        state = mol_to_feature_vector(None)
        trajectory = []
        terminated = False
        
        while not terminated and mol.GetNumAtoms() < MAX_LENGTH:
            # Get action probabilities from policy network
            state_input = state.unsqueeze(0)  # Add batch dimension
            action_probs = policy_net(state_input)
            action_probs = action_probs.squeeze(0)  # Remove batch dimension
            action_dist = torch.distributions.Categorical(action_probs)
            action_idx = action_dist.sample()
            action = ACTIONS[action_idx]
            
            trajectory.append((state, action_idx))
            
            if action == 'Terminate':
                terminated = True
            else:
                # Add atom to the molecule
                atom = Chem.Atom(action)
                mol.AddAtom(atom)
                # For simplicity, we won't add bonds
                
                # Update state
                state = mol_to_feature_vector(mol)
        
        # Compute reward
        final_mol = mol.GetMol()
        reward = compute_reward(final_mol)
        
        # Backpropagation
        loss = 0.0
        R = reward
        for state, action_idx in reversed(trajectory):
            state_input = state.unsqueeze(0)  # Add batch dimension
            action_probs = policy_net(state_input)
            action_probs = action_probs.squeeze(0)  # Remove batch dimension
            log_prob = torch.log(action_probs[action_idx] + 1e-10)  # Avoid log(0)
            loss += -log_prob * R
            R *= gamma  # Apply discount factor
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item():.4f}, Reward: {reward:.4f}")

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

# Train the GFlowNet
train_gflownet()

# Sample molecules
sampled_molecules = []
for _ in range(20):
    mol = sample_molecule()
    sampled_molecules.append(mol)

# Filter out None molecules
sampled_molecules = [mol for mol in sampled_molecules if mol is not None and mol.GetNumAtoms() > 0]

# Visualize the molecules
img = MolsToGridImage(sampled_molecules, molsPerRow=5, subImgSize=(200,200))
img.show()
