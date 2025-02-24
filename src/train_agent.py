def train_agent(args, config, n_episodes=1000):
    # ... (previous initialization code) ...
    
    # Initialize agent with new state size
    state_size = 8  # Updated to match new state representation
    action_size = 3
    agent = TradingAgent(
        state_size=state_size,
        action_size=action_size,
        config=config,
        learning_rate=0.0005,
        gamma=0.95,
        epsilon_decay=0.995,
        memory_size=20000
    )
    
    # ... (rest of the code remains the same) ... 