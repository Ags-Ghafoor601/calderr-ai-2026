# Reinforcement Learning

Reinforcement learning (RL) is a paradigm where agents learn optimal behavior through trial and error.

## Key Concepts
- **Agent**: The learner or decision-maker
- **Environment**: The world the agent interacts with
- **State**: Current situation of the agent
- **Action**: What the agent can do
- **Reward**: Feedback signal indicating how good an action was
- **Policy**: Strategy that the agent follows (maps states to actions)

## Algorithms

### Q-Learning
A model-free algorithm that learns the value of state-action pairs.
The Q-table stores expected cumulative rewards for each state-action combination.

### Deep Q-Networks (DQN)
Combines Q-learning with deep neural networks to handle large state spaces.
Used by DeepMind to achieve superhuman performance in Atari games.

### Policy Gradient Methods
Directly optimize the policy function rather than the value function.
REINFORCE algorithm uses the policy gradient theorem.

### Actor-Critic Methods
Combine value-based and policy-based approaches.
A2C, A3C, and PPO (Proximal Policy Optimization) are popular variants.

## Applications
- Game AI (AlphaGo, OpenAI Five)
- Robotics (manipulation, locomotion)
- Resource management and scheduling
- Recommendation systems