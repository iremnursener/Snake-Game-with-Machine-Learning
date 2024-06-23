# Snake-Game-with-Machine-Learning
In this project, the goal is for an agent to play the classic Snake game using reinforcement learning. In this study, the agent acts as a snake and collects food in the game area. Each time the agent collects food, it grows and optimizes itself to collect more food in the game area.
## Languages Used
-Phyton 


Project Description
In this project, the goal is for an agent to play the classic Snake game using reinforcement learning. The agent moves based on inputs from its environment and aims to reach a specific target. In this work, the agent acts as a snake and collects food items in the game area. Each time the agent collects a food item, it grows and optimizes itself to collect more food in the game area. The agent's goal is to achieve the highest score and survive as long as possible.

Research (Preliminary Work)
Before starting the project, a comprehensive literature review on reinforcement learning, Q-learning, and deep learning algorithms was conducted. Algorithms used in Snake and similar games were examined.
Sources:
• Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.
• Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature.
• Similar projects and code examples on GitHub were reviewed.
• YouTube videos and Udemy courses were utilized.

Environment, Methods, and Libraries Used

torch: PyTorch library for deep learning.
torch.nn: Submodule for defining neural network architectures.
torch.optim: Submodule for optimization algorithms.
torch.nn.functional: Submodule for common activation functions and loss functions.
os: Library for file system operations.
matplotlib.pyplot: Library for creating graphs.
IPython.display: Library for displaying graphs in Jupyter Notebook.
pygame: Library for game development (used in the Snake Game environment).
random: Library for generating random numbers.
enum: Library for defining constants.
collections: Library for data structures like Namedtuples.
numpy: Library for numerical computations.
IDE: PyCharm


Proposed (Developed / Used) Method
-------------------------------------
The method used in our project to develop an AI-powered Snake game is the Q-learning algorithm.
Q-Learning and Deep Q-Networks (DQN)
Q-learning is a type of reinforcement learning algorithm. The agent evaluates a state, takes actions, and improves its future actions based on the rewards received from these actions.
Deep Q-Networks (DQN) is the integration of Q-learning algorithm with deep learning. Instead of a traditional Q-learning table, a neural network is used to predict Q-values.
Neural Network (NN): A neural network model used to predict Q-values. This model is created using PyTorch.
Experience Replay: This technique is used to make the learning process more efficient by utilizing the agent's past experiences. It involves randomly sampling from the memory (deque) of past experiences.

Agent Class
------------
The Agent class contains all necessary methods for the agent to play and learn the game.
• Memory: A deque data structure is used to store the agent's past experiences. This enables experience replay.
• Model and Training: A neural network (Linear_QNet) is used to predict Q-values and a training class (QTrainer) is used to train this network.
• State Representation: The game state is represented by an 11-dimensional vector, which includes information about the agent's surroundings and the position of the food.
• Action Selection: The agent uses the epsilon-greedy strategy to select actions. This means that with a certain probability, the agent selects random actions (exploration) and follows the learned policy at other times (exploitation).
• Short and Long Term Memory Training: The agent uses short-term memory to learn at each step and long-term memory at certain intervals.

Game Class
------------
The SnakeGameAI class contains the game logic and graphical interface (pygame).
• Game State and Reset: The game state is initialized and can be reset.
• Food Placement: Food is placed at random locations, but not on the snake.
• Game Step: At each step, the agent performs an action, receives the new state and reward, and checks if the game is over.
• Collision Check: Checks if the snake has collided with the walls or its own body.
• UI Update: Updates the game's visual interface.

Model and Training
--------------------
The Linear_QNet and QTrainer classes define and train the deep learning model.
• Linear_QNet: A simple two-layer neural network. It consists of an input layer, a hidden layer, and an output layer.
• QTrainer: Defines the necessary optimization and loss functions for training the model. It also includes steps to update Q-values.
Visualization
The plot function graphically shows the scores and average scores during training to monitor the agent's performance.

Methods Used:
-----------------
• Epsilon-Greedy Strategy: Used to balance exploration and exploitation.
• Experience Replay: Training is done using randomly sampled experiences from memory, which makes learning more stable.
• Mini-batch Learning: Instead of using all examples in memory during training, a small subset is used.
• Loss Function: Mean Squared Error (MSE) loss function is used to accurately predict Q-values.
• Optimization: Adam optimizer is used to update the model parameters.

Experimental Studies
----------------------
Experimental studies focus on agent training and performance evaluation:
Agent Training: The agent is continuously trained with data collected during the game. This process is carried out using both short-term and long-term memory.
Model Performance: The performance of the model is evaluated using scores and average scores in the game. Scores and average scores are visualized using Matplotlib. As the program runs, it is observed that the score increases and it is visualized in a table.

Conclusion
------------
In this project, an agent was developed for the Snake game using reinforcement learning algorithms. By the end of the training process, the agent successfully learned to collect food and survive. The results obtained demonstrate the effectiveness and applicability of the method.


Appendix 1: Performance Improvement
--------------------------------------
Performance improvement focuses on optimizing the learning process of the model:
Experience Replay: Reuse of past experiences for more efficient learning.
Epsilon Greedy Strategy: Balancing exploration and exploitation to achieve better performance.

![image](https://github.com/iremnursener/Snake-Game-with-Machine-Learning/assets/119794427/1b6f7e32-5e6a-486f-acc2-be7ebe4eedc9)





