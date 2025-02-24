import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import logging

logger = logging.getLogger(__name__)

class TradingAgent:
    def __init__(self, state_size, action_size, config, learning_rate=0.0001, 
                 gamma=0.99, epsilon_decay=0.999, memory_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = epsilon_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.update_target_every = 100
        self.target_update_counter = 0
        
    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 512),
            nn.LeakyReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
            nn.Linear(128, self.action_size)
        )
        return model
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience with priority"""
        # Calculate priority based on reward magnitude
        priority = abs(reward) + 0.01  # Small constant to ensure non-zero priority
        self.memory.append((state, action, reward, next_state, done, priority))
        
        # Keep memory size in check
        if len(self.memory) > self.memory.maxlen:
            # Remove lowest priority experience
            min_priority_idx = min(range(len(self.memory)), 
                                 key=lambda i: self.memory[i][5])
            del self.memory[min_priority_idx]
        
    def act(self, state):
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Add temperature-based exploration
        if random.random() <= self.epsilon:
            # Use softmax exploration instead of pure random
            temperature = max(0.1, self.epsilon)  # Higher temp = more random
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state)[0]
                probs = torch.softmax(q_values / temperature, dim=0)
                return np.random.choice(self.action_size, p=probs.cpu().numpy())
            
        # Greedy action
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            act_values = self.model(state)
            return np.argmax(act_values[0].cpu().numpy())
            
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        # Sort experiences by absolute reward
        sorted_memory = sorted(self.memory, key=lambda x: abs(x[2]), reverse=True)
        
        # Calculate sizes for top and random samples
        top_size = min(int(len(sorted_memory) * 0.2), batch_size // 2)
        random_size = min(batch_size - top_size, len(sorted_memory) - top_size)
        
        # Get top experiences
        top_experiences = sorted_memory[:top_size]
        
        # Get random experiences from remaining memory
        if random_size > 0:
            random_experiences = random.sample(sorted_memory[top_size:], random_size)
        else:
            random_experiences = []
        
        # Combine experiences
        minibatch = top_experiences + random_experiences
        
        if len(minibatch) == 0:
            return
        
        # Process batch
        states = torch.FloatTensor([t[0] for t in minibatch]).to(self.device)
        actions = torch.LongTensor([t[1] for t in minibatch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in minibatch]).to(self.device)
        next_states = torch.FloatTensor([t[3] for t in minibatch]).to(self.device)
        dones = torch.FloatTensor([t[4] for t in minibatch]).to(self.device)
        
        # Double DQN update
        with torch.no_grad():
            # Get actions from main network
            next_actions = self.model(next_states).max(1)[1].unsqueeze(1)
            # Get Q-values from target network for those actions
            next_q = self.target_model(next_states).gather(1, next_actions)
            # Calculate target Q-values
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q
        
        # Get current Q-values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Calculate loss with Huber Loss (more robust than MSE)
        loss = nn.HuberLoss()(current_q, target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Prevent exploding gradients
        self.optimizer.step()
        
        # Update target network periodically
        self.target_update_counter += 1
        if self.target_update_counter >= self.update_target_every:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0
        
        return loss.item()  # Return loss for monitoring

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load_model(self, path):
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            logger.info("Successfully loaded model from %s", path)
        except Exception as e:
            logger.warning(f"Error loading model: {e}. Starting with fresh model.") 