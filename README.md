A simple custom  Gymnasium environment implementing a simple grid-world where an agent learns to navigate to a target position using Proximal Policy Optimization (PPO).

## Overview

This project implements a grid-based environment where:
- An agent (A) navigates a customizable NxN grid
- A target (T) is randomly placed on the grid
- The agent must learn to reach the target using reinforcement learning
- When the agent reaches the target, they merge into a goal state (G)

## Environment Details

### Observation Space
The environment uses a Dictionary observation space containing:
- `agent`: Box(low=0, high=size-1, shape=(2,), dtype=int)
  - Represents agent's (x,y) position on the grid
  - Both coordinates are bounded between 0 and grid size-1
- `target`: Box(low=0, high=size-1, shape=(2,), dtype=int)
  - Represents target's (x,y) position on the grid
  - Both coordinates are bounded between 0 and grid size-1

### Action Space
- `Discrete(4)`: Four possible actions
  - 0: Move Right [1, 0]
  - 1: Move Up [0, 1]
  - 2: Move Left [-1, 0]
  - 3: Move Down [0, -1]

### Reward Structure
- +1: Agent reaches the target position
- 0: All other actions
- No negative rewards or step penalties

### Episode Termination
Episodes can end under two conditions:
- `terminated=True`: Agent successfully reaches the target
- `truncated=False`: No truncation implemented

### State Transitions
- Agent movement is bounded by grid walls (using np.clip)
- Initial state:
  - Agent spawns at random position
  - Target spawns at random position different from agent
- Manhattan distance between agent and target is tracked in info dict

### Additional Information
The `info` dictionary contains:
- `distance`: Manhattan distance (L1 norm) between agent and target

### Reset Behavior
On reset:
1. Agent is placed at random position
2. Target is placed at random position (ensuring different from agent)
3. Returns initial observation and info dictionary
4. Supports optional seed for reproducibility

### Rendering
ASCII grid visualization where:
- `A`: Agent's current position
- `T`: Target position
- `G`: Goal state (when agent reaches target)
- `*`: Empty cell
