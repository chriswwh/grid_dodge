import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
import onnx # Import the onnx library directly
import random
import math
from collections import deque
import matplotlib.pyplot as plt

# ==========================================
# 1. THE ENVIRONMENT (Game Logic Port)
# ==========================================
class GridDodgeEnv:
    def __init__(self):
        # Constants matching your JS
        self.grid_size = 2
        self.cell_size = 100
        # Normalize coordinates so the grid is roughly 0.0 to 2.0
        self.scale = 100.0
        # Match HTML's INIT_SPIKE_SPEED
        self.init_spike_speed = 10
        # Training will cover speeds from 5 to 50 by varying score

        self.reset()

    def reset(self):
        self.player_pos = [0, 0] # Grid coordinates [x, y] (0 or 1)
        self.hazards = []
        self.score = 0
        self.spawn_timer = 0
        self.steps_alive = 0
        return self._get_state()

    def _spawn_hazard(self):
        # Mimics the JS spawnHazard logic (2 triangles)
        used_direction = -1

        for _ in range(2):
            target_x = random.randint(0, 1)
            target_y = random.randint(0, 1)

            # Pick distinct directions
            direction = random.randint(0, 3)
            while direction == used_direction:
                direction = random.randint(0, 3)
            used_direction = direction

            # Setup Start/Target in simulation units
            # Grid centers are at 0.5, 1.5 (if we treat cell size as 1.0)
            tx_center = target_x + 0.5
            ty_center = target_y + 0.5

            if direction == 0:   # Top
                sx, sy = tx_center, -0.5
            elif direction == 1: # Right
                sx, sy = 2.5, ty_center
            elif direction == 2: # Bottom
                sx, sy = tx_center, 2.5
            else:                # Left
                sx, sy = -0.5, ty_center

            # Speed calculation (Match the HTML game logic exactly)
            # getGameSpeed() = INIT_SPIKE_SPEED * (1 + (score / 30) * 0.2)
            # Scaled to normalized coordinates (divide by cell_size)
            base_speed = (self.init_spike_speed / self.cell_size) * (1 + (self.score / 30) * 0.2)
            variance = 0.8 + (random.random() * 0.4)  # 0.8 to 1.2
            speed = base_speed * variance

            # Velocity vector
            angle = math.atan2(ty_center - sy, tx_center - sx)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed

            self.hazards.append({
                'x': sx, 'y': sy,
                'vx': vx, 'vy': vy,
                'tx': target_x, 'ty': target_y,
                'passed': False
            })

    def step(self, action):
        # Action: 0=Up, 1=Right, 2=Down, 3=Left, 4=Wait
        move_map = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

        reward = 0.1 # Survival reward
        done = False

        # 1. Move Player
        if action in move_map:
            dx, dy = move_map[action]
            nx = self.player_pos[0] + dx
            ny = self.player_pos[1] + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                self.player_pos = [nx, ny]

        # 2. Update Hazards
        active_hazards = []
        wave_cleared = True

        for h in self.hazards:
            h['x'] += h['vx']
            h['y'] += h['vy']

            # Boundary removal (bounds are -1 to 3 roughly)
            if -1.0 < h['x'] < 3.0 and -1.0 < h['y'] < 3.0:
                active_hazards.append(h)
                if not h['passed']:
                    wave_cleared = False

            # Collision Check
            # Player center
            px = self.player_pos[0] + 0.5
            py = self.player_pos[1] + 0.5
            dist = math.hypot(h['x'] - px, h['y'] - py)

            # Hitbox radius approx 0.3 in simulation units
            if dist < 0.25:
                reward = -10 # Death penalty
                done = True

            # Scoring (Passed target)
            tx_center = h['tx'] + 0.5
            ty_center = h['ty'] + 0.5
            dist_to_target = math.hypot(h['x'] - tx_center, h['y'] - ty_center)

            if not h['passed'] and dist_to_target < 0.2: # Passed threshold
                h['passed'] = True
                self.score += 1
                reward += 1.0 # Bonus for dodging

        self.hazards = active_hazards

        # 3. Spawn Logic
        self.spawn_timer += 1
        if len(self.hazards) == 0 or wave_cleared:
             if self.spawn_timer > 10: # Small delay
                self._spawn_hazard()
                self.spawn_timer = 0

        self.steps_alive += 1
        return self._get_state(), reward, done

    def _get_state(self):
        # We need a fixed input size for the Neural Network.
        # We will track the player and the 2 closest hazards.

        state = [
            self.player_pos[0], # Player X
            self.player_pos[1], # Player Y
        ]

        # Sort hazards by distance to player
        px, py = self.player_pos[0] + 0.5, self.player_pos[1] + 0.5
        sorted_hazards = sorted(self.hazards, key=lambda h: math.hypot(h['x']-px, h['y']-py))

        # Take up to 2 nearest hazards (since game spawns 2 at a time)
        # Input per hazard: [rel_x, rel_y, vx, vy]
        for i in range(2):
            if i < len(sorted_hazards):
                h = sorted_hazards[i]
                state.extend([
                    h['x'] - px, # Relative X
                    h['y'] - py, # Relative Y
                    h['vx'] * 10, # Scale up small velocity numbers
                    h['vy'] * 10
                ])
            else:
                # Padding if no hazard
                state.extend([0, 0, 0, 0])

        return np.array(state, dtype=np.float32)

# ==========================================
# 2. DEEP Q-NETWORK (The Brain)
# ==========================================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train():
    env = GridDodgeEnv()

    # State: Player(2) + Hazard1(4) + Hazard2(4) = 10 inputs
    # Actions: Up, Right, Down, Left, Wait = 5 outputs
    model = DQN(10, 5)
    target_model = DQN(10, 5)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    replay_buffer = deque(maxlen=10000)

    epsilon = 1.0
    epsilon_decay = 0.9995
    epsilon_min = 0.05
    batch_size = 64
    gamma = 0.99

    episodes = 5000
    scores = []

    print("Starting Training...")

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Epsilon-Greedy Action Selection
            if random.random() < epsilon:
                action = random.randint(0, 4)
            else:
                with torch.no_grad():
                    q_values = model(torch.tensor(state).unsqueeze(0))
                    action = torch.argmax(q_values).item()

            # Step
            next_state, reward, done = env.step(action)

            # Store transition
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # Train
            if len(replay_buffer) > batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor(np.array(states))
                actions = torch.tensor(actions).unsqueeze(1)
                rewards = torch.tensor(rewards).unsqueeze(1)
                next_states = torch.tensor(np.array(next_states))
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

                # Q(s, a)
                current_q = model(states).gather(1, actions)

                # R + gamma * max(Q(s', a'))
                with torch.no_grad():
                    max_next_q = target_model(next_states).max(1)[0].unsqueeze(1)
                    target_q = rewards + (1 - dones) * gamma * max_next_q

                loss = loss_fn(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update Target Network
        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())

        # Decay Epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        scores.append(total_reward)

        if episode % 50 == 0:
            avg_score = np.mean(scores[-50:])
            print(f"Episode {episode}, Avg Reward: {avg_score:.2f}, Epsilon: {epsilon:.2f}")

    # Save the model
    torch.save(model.state_dict(), "grid_dodge_dqn.pth")
    print("Training Complete. Model saved.")

    export_model()

    # Plot results
    # plt.plot(scores)
    # plt.title('Training Performance')
    # plt.xlabel('Episode')
    # plt.ylabel('Total Reward')
    # plt.show()

def export_model():
    INPUT_SIZE = 10
    OUTPUT_SIZE = 5
    PATH_TO_PTH = "grid_dodge_dqn.pth"
    PATH_TO_ONNX = "grid_dodge_dqn.onnx"

    print(f"Loading model from {PATH_TO_PTH}...")

    model = DQN(INPUT_SIZE, OUTPUT_SIZE)
    try:
        model.load_state_dict(torch.load(PATH_TO_PTH, map_location='cpu'))
    except FileNotFoundError:
        print(f"Error: Could not find {PATH_TO_PTH}")
        return

    model.eval()
    dummy_input = torch.randn(1, INPUT_SIZE)

    print(f"Exporting to {PATH_TO_ONNX}...")

    # Export to a temporary file first
    torch.onnx.export(
        model,
        dummy_input,
        "temp_model.onnx", # Temp file
        export_params=True,
        opset_version=10,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    # RELOAD AND SAVE AS SINGLE FILE (The Fix)
    # This forces ONNX to bundle weights inside the main file
    onnx_model = onnx.load("temp_model.onnx")
    onnx.save_model(
        onnx_model,
        PATH_TO_ONNX,
        save_as_external_data=False
    )

    print("Success! Created a self-contained 'grid_dodge.onnx'.")

if __name__ == "__main__":
    train()