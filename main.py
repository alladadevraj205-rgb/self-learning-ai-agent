import random
import json
import os

# ==========================================
# SELF-LEARNING AI AGENT
# Reinforcement Learning using Q-Learning
# ==========================================

GRID_SIZE = 5

START = (0, 0)
GOAL = (4, 4)

OBSTACLES = [
    (1, 1),
    (2, 1),
    (3, 3)
]

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

# Learning settings
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

EPISODES = 5000
MAX_STEPS = 100

MODEL_FILE = "q_table.json"

# Q-table stores learned knowledge
q_table = {}


# Convert state and action into a dictionary key
def make_key(state, action):
    return f"{state[0]},{state[1]},{action}"


# Get the learned value of an action
def get_q(state, action):
    return q_table.get(make_key(state, action), 0.0)


# Find valid movements
def get_valid_actions(state):

    x, y = state

    moves = {
        "UP": (x - 1, y),
        "DOWN": (x + 1, y),
        "LEFT": (x, y - 1),
        "RIGHT": (x, y + 1)
    }

    valid_actions = []

    for action, position in moves.items():

        nx, ny = position

        if (
            0 <= nx < GRID_SIZE
            and 0 <= ny < GRID_SIZE
            and position not in OBSTACLES
        ):
            valid_actions.append(action)

    return valid_actions


# Move the agent
def move(state, action):

    x, y = state

    if action == "UP":
        x -= 1

    elif action == "DOWN":
        x += 1

    elif action == "LEFT":
        y -= 1

    elif action == "RIGHT":
        y += 1

    new_state = (x, y)

    if (
        x < 0 or x >= GRID_SIZE
        or y < 0 or y >= GRID_SIZE
        or new_state in OBSTACLES
    ):
        return state

    return new_state


# Choose an action using exploration or learning
def choose_action(state, epsilon):

    valid_actions = get_valid_actions(state)

    if not valid_actions:
        return None

    # Exploration
    if random.random() < epsilon:
        return random.choice(valid_actions)

    # Exploitation
    best_value = max(
        get_q(state, action)
        for action in valid_actions
    )

    best_actions = [
        action for action in valid_actions
        if get_q(state, action) == best_value
    ]

    return random.choice(best_actions)


# Train the AI agent
def train_agent():

    global EPSILON

    print("\nAI training started...\n")

    for episode in range(1, EPISODES + 1):

        state = START
        total_reward = 0

        for step in range(MAX_STEPS):

            action = choose_action(state, EPSILON)

            if action is None:
                break

            next_state = move(state, action)

            # Reward system
            if next_state == GOAL:
                reward = 100

            elif next_state == state:
                reward = -10

            else:
                reward = -1

            old_value = get_q(state, action)

            # Calculate future reward
            next_actions = get_valid_actions(next_state)

            if next_state == GOAL or not next_actions:
                future_value = 0

            else:
                future_value = max(
                    get_q(next_state, a)
                    for a in next_actions
                )

            # Q-Learning update formula
            new_value = old_value + LEARNING_RATE * (
                reward
                + DISCOUNT_FACTOR * future_value
                - old_value
            )

            q_table[make_key(state, action)] = new_value

            state = next_state
            total_reward += reward

            if state == GOAL:
                break

        # Reduce exploration over time
        EPSILON = max(
            MIN_EPSILON,
            EPSILON * EPSILON_DECAY
        )

        if episode % 500 == 0:

            print(
                f"Episode: {episode}/{EPISODES} | "
                f"Exploration: {EPSILON:.3f} | "
                f"Reward: {total_reward}"
            )

    print("\nTraining completed successfully!")


# Save learned knowledge
def save_model():

    with open(MODEL_FILE, "w") as file:
        json.dump(q_table, file, indent=4)

    print("Learned knowledge saved to", MODEL_FILE)


# Load previously learned knowledge
def load_model():

    global q_table

    if os.path.exists(MODEL_FILE):

        try:
            with open(MODEL_FILE, "r") as file:
                q_table = json.load(file)

            print("Previous learning loaded!")

        except (json.JSONDecodeError, OSError):
            q_table = {}
            print("Starting with a fresh Q-table.")

    else:
        print("No previous model found. Starting fresh.")


# Display the environment
def show_grid(path=None):

    path = path or []

    print()

    for x in range(GRID_SIZE):

        row = ""

        for y in range(GRID_SIZE):

            position = (x, y)

            if position == START:
                symbol = "S"

            elif position == GOAL:
                symbol = "G"

            elif position in OBSTACLES:
                symbol = "#"

            elif position in path:
                symbol = "*"

            else:
                symbol = "."

            row += symbol + " "

        print(row)


# Display the learned path
def show_learned_path():

    state = START
    path = [state]
    visited = {state}

    for step in range(MAX_STEPS):

        if state == GOAL:
            break

        action = choose_action(state, 0)

        if action is None:
            break

        next_state = move(state, action)

        if next_state == state or next_state in visited:
            break

        state = next_state
        path.append(state)
        visited.add(state)

    print("\nAI's learned path:")

    show_grid(path)

    if state == GOAL:

        print("\nSUCCESS! AI reached its destination.")
        print("Total steps:", len(path) - 1)

    else:

        print("\nAI could not find a complete path.")


# Main program
if __name__ == "__main__":

    print("=" * 40)
    print("       SELF-LEARNING AI AGENT")
    print("=" * 40)

    print("\nInitial environment:")
    show_grid()

    load_model()

    train_agent()

    save_model()

    show_learned_path()

    print("\nThank you for running the AI Agent!")