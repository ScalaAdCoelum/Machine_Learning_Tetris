import pygame
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import pyautogui
import os

screenshot_dir = "screenshots"
os.makedirs(screenshot_dir, exist_ok=True)

# Constants
WIDTH, HEIGHT = 300, 600
GRID_SIZE = 30
GRID_WIDTH, GRID_HEIGHT = WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE
WHITE, BLACK = (255, 255, 255), (0, 0, 0)


best_fitness_scores = []

# Tetris shapes with corresponding colors
SHAPES = {
    "I": {"shape": [[1, 1, 1, 1]], "color": (0, 255, 255)},
    "O": {"shape": [[1, 1], [1, 1]], "color": (255, 255, 0)},
    "T": {"shape": [[1, 1, 1], [0, 1, 0]], "color": (128, 0, 128)},
    "L": {"shape": [[1, 1, 1], [1, 0, 0]], "color": (255, 165, 0)},
    "J": {"shape": [[1, 1, 1], [0, 0, 1]], "color": (0, 0, 255)},
    "S": {"shape": [[1, 1, 0], [0, 1, 1]], "color": (0, 255, 0)},
    "Z": {"shape": [[0, 1, 1], [1, 1, 0]], "color": (255, 0, 0)}
}

# Scoring constants
LINE_CLEAR_POINTS = {1: 100, 2: 300, 3: 500, 4: 800}  # Points for clearing 1, 2, 3, or 4 lines
BACK_TO_BACK_TETRIS_BONUS = 1200  # Bonus for consecutive Tetris clears

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tetris")

# Functions
def draw_next_piece(next_shape, color):
    next_piece_size = 4 
    next_piece_x = WIDTH + 20
    next_piece_y = 20

    for row in range(len(next_shape)):
        for col in range(len(next_shape[row])):
            if next_shape[row][col] == 1:
                pygame.draw.rect(
                    screen,
                    color,
                    (
                        next_piece_x + col * GRID_SIZE,
                        next_piece_y + row * GRID_SIZE,
                        GRID_SIZE,
                        GRID_SIZE,
                    ),
                )

def draw_grid():
    for x in range(0, WIDTH, GRID_SIZE):
        pygame.draw.line(screen, WHITE, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, WHITE, (0, y), (WIDTH, y))

def draw_shape(shape, x, y, color):
    for row in range(len(shape)):
        for col in range(len(shape[row])):
            if shape[row][col] == 1:
                pygame.draw.rect(screen, color, (x + col * GRID_SIZE, y + row * GRID_SIZE, GRID_SIZE, GRID_SIZE))

def rotate_shape(shape):
    return [list(row) for row in zip(*reversed(shape))]

def check_collision(shape, x, y):
    for row in range(len(shape)):
        for col in range(len(shape[row])):
            if shape[row][col] == 1 and (x + col < 0 or x + col >= GRID_WIDTH or y + row >= GRID_HEIGHT or grid[y + row][x + col]):
                return True
    return False

def check_t_spin(grid, shape, x, y):
    # Check if the current shape is a T-shaped Tetrimino
    if shape != SHAPES["T"]["shape"]:
        return False

    # Check for T-Spin opportunities
    t_spin_conditions = [
        (0, 0), (0, 2), (2, 0), (2, 2)
    ]

    for condition in t_spin_conditions:
        dx, dy = condition
        if (
            grid[y + dy][x + dx] != 0
            or grid[y + dy + 1][x + dx] != 0
            or grid[y + dy][x + dx + 1] != 0
            or grid[y + dy + 1][x + dx + 1] != 0
        ):
            return False

    return True

def execute_t_spin(grid, shape, x, y):
    # Execute T-Spin by clearing lines
    for row in range(len(shape)):
        for col in range(len(shape[row])):
            if shape[row][col] == 1:
                grid[y + row][x + col] = 0

    # Clear any lines
    clear_rows()

def check_perfect_clear(grid):
    # Check if the grid has no blocks (Perfect Clear condition)
    return np.all(grid == 0)

def execute_perfect_clear(grid):
    # Clear all lines to achieve Perfect Clear
    for row in range(GRID_HEIGHT):
        grid[row] = [0] * GRID_WIDTH

def clear_rows():
    global grid, score
    full_rows = [row for row in range(GRID_HEIGHT) if all(grid[row])]
    lines_cleared = len(full_rows)

    if lines_cleared > 0:
        # Calculate score based on the number of lines cleared
        score += LINE_CLEAR_POINTS.get(lines_cleared, 0)

        # Check for back-to-back Tetris bonus
        if lines_cleared == 4 and last_lines_cleared == 4:
            score += BACK_TO_BACK_TETRIS_BONUS

        last_lines_cleared = lines_cleared

        for row in full_rows:
            del grid[row]
            grid.insert(0, [0] * GRID_WIDTH)

# === Those functions were found by checking the Tetris Wiki where all the technics are indexed.
# === The functions are used to calculate the heuristic values for the genetic algorithm.

# Genetic Algorithm Constants
POPULATION_SIZE = 500
GENERATIONS = 100
    
# Define Heuristic Weights
FITNESS_WEIGHTS = {
    "Height": 2,
    "Lines Cleared": 5,
    "Holes": 10,
    "Bumpiness": 2,
    "T-Spin": 50,
    "Blockades": 20,
    "Wells": 10,
    "Connected Holes": 10
}

# Heuristic Model
def heuristic_model(grid, current_shape, shape_x, shape_y):
    height_penalty = calculate_height_penalty(grid)
    lines_cleared_bonus = calculate_lines_cleared_bonus(grid)
    holes_penalty = calculate_holes_penalty(grid)
    bumpiness_penalty = calculate_bumpiness_penalty(grid)
    t_spin_bonus = calculate_t_spin_bonus(grid, current_shape, shape_x, shape_y)
    blockades_penalty = calculate_blockades_penalty(grid)
    wells_penalty = calculate_wells_penalty(grid)
    connected_holes_penalty = calculate_connected_holes_penalty(grid)
    empty_space_reward = calculate_empty_space_reward(grid, current_shape, shape_x, shape_y)


    return (
        200 * height_penalty +
        500 * lines_cleared_bonus +
        10 * holes_penalty +
        2 * bumpiness_penalty +
        10 * t_spin_bonus -
        20 * blockades_penalty -
        10 * wells_penalty -
        100 * connected_holes_penalty +
        empty_space_reward
    )

def calculate_empty_space_reward(grid, current_shape, shape_x, shape_y):
    reward = 0
    for row in range(len(current_shape)):
        for col in range(len(current_shape[row])):
            if current_shape[row][col] == 1:
                target_row = shape_y + row
                target_col = shape_x + col
                # Check if the target position is an empty space or a hole
                if target_row < GRID_HEIGHT and target_col < GRID_WIDTH and grid[target_row][target_col] == 0:
                    reward += 1000
    return reward

def calculate_blockades_penalty(grid):
    blockades = 0
    for col in range(GRID_WIDTH):
        for row in range(GRID_HEIGHT - 1):
            if grid[row][col] != 0 and grid[row + 1][col] == 0:
                blockades += 1
    return blockades

def calculate_wells_penalty(grid):
    wells = 0
    for col in range(1, GRID_WIDTH - 1):
        for row in range(GRID_HEIGHT):
            if grid[row][col] == 0 and grid[row][col - 1] != 0 and grid[row][col + 1] != 0:
                well_depth = 1
                while row + well_depth < GRID_HEIGHT and grid[row + well_depth][col] == 0:
                    well_depth += 1
                wells += well_depth
    return wells

def calculate_connected_holes_penalty(grid):
    connected_holes = 0
    for col in range(1, GRID_WIDTH - 1):
        for row in range(1, GRID_HEIGHT - 1):
            if (
                grid[row][col] == 0
                and grid[row - 1][col] != 0
                and grid[row + 1][col] != 0
                and grid[row][col - 1] != 0
                and grid[row][col + 1] != 0
            ):
                connected_holes += 1
    return connected_holes

def calculate_height_penalty(grid):
    heights = [0] * GRID_WIDTH
    for col in range(GRID_WIDTH):
        for row in range(GRID_HEIGHT):
            if grid[row][col] != 0:
                heights[col] = GRID_HEIGHT - row
                break
    return sum(heights)

def calculate_lines_cleared_bonus(grid):
    lines_cleared = GRID_HEIGHT - len([row for row in grid if 0 not in row])
    return LINE_CLEAR_POINTS.get(lines_cleared, 0)

def calculate_holes_penalty(grid):
    holes = 0
    for col in range(GRID_WIDTH):
        block_found = False
        for row in range(GRID_HEIGHT):
            if grid[row][col] != 0:
                block_found = True
            elif block_found:
                holes += 1
    return holes

def calculate_bumpiness_penalty(grid):
    heights = [0] * GRID_WIDTH
    for col in range(GRID_WIDTH):
        for row in range(GRID_HEIGHT):
            if grid[row][col] != 0:
                heights[col] = GRID_HEIGHT - row
                break

    bumpiness = 0
    for i in range(len(heights) - 1):
        bumpiness += abs(heights[i] - heights[i + 1])
    return bumpiness

def calculate_t_spin_bonus(grid, current_shape, shape_x, shape_y):
    if check_t_spin(grid, current_shape, shape_x, shape_y):
        return BACK_TO_BACK_TETRIS_BONUS
    return 0

#Pi = c-ri / c*(c-1)

def rank_based_selection(population):
    n = len(population)
    ranks = np.argsort([-ind['fitness'] for ind in population])
    probabilities = [(2.0 - s) / n for s in range(n)]
    
    # Ensure probabilities are non-negative
    probabilities = np.maximum(probabilities, 0)
    
    # Normalize probabilities to sum up to 1
    probabilities /= probabilities.sum()

    selected_indices = np.random.choice(ranks, size=n, p=probabilities)
    selected_individuals = [population[i] for i in selected_indices]
    return selected_individuals

# Evaluate_agent function to use the heuristic model
def evaluate_agent(agent, generation):
    global grid, score, last_lines_cleared, fall_speed

    # Reset the game state
    reset_game()

    fall_speed = 500

    screenshot_interval = 1000
    screenshot_counter = 0

    best_agent_fitness = float('-inf')
    best_agent_generation_fitness = 0  # Track the fitness of the best agent within the current generation

    for _ in range(GENERATIONS):
        current_shape_name = random.choice(list(SHAPES.keys()))
        current_shape = SHAPES[current_shape_name]["shape"]
        shape_color = SHAPES[current_shape_name]["color"]

        shape_x, shape_y = GRID_WIDTH // 2 - len(current_shape[0]) // 2, 0
        fall_time = 0
        running = True

        if current_shape_name == "I" or current_shape_name == "T":
            mound_height = GRID_HEIGHT - 4
            shape_y = mound_height

        while running:
            keys = [random.choice([pygame.K_LEFT, pygame.K_RIGHT, pygame.K_DOWN, pygame.K_SPACE])]
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            for key in keys:
                if key == pygame.K_SPACE:
                    rotated_shape = rotate_shape(current_shape)
                    if not check_collision(rotated_shape, shape_x, shape_y):
                        current_shape = rotated_shape

                if key == pygame.K_LEFT and not check_collision(current_shape, shape_x - 1, shape_y):
                    shape_x -= 1
                if key == pygame.K_RIGHT and not check_collision(current_shape, shape_x + 1, shape_y):
                    shape_x += 1
                if key == pygame.K_DOWN and not check_collision(current_shape, shape_x, shape_y + 1):
                    shape_y += 1

            now = pygame.time.get_ticks()
            if now - fall_time > fall_speed:
                if not check_collision(current_shape, shape_x, shape_y + 1):
                    shape_y += 1
                else:
                    for row in range(len(current_shape)):
                        for col in range(len(current_shape[row])):
                            if current_shape[row][col] == 1:
                                grid[shape_y + row][shape_x + col] = shape_color
                    clear_rows()

                    if any(grid[0]):
                        running = False

                    shape_x, shape_y = GRID_WIDTH // 2 - len(current_shape[0]) // 2, 0
                    current_shape_name = random.choice(list(SHAPES.keys()))
                    current_shape = SHAPES[current_shape_name]["shape"]
                    shape_color = SHAPES[current_shape_name]["color"]

                    if current_shape_name == "I" or current_shape_name == "T":
                        mound_height = GRID_HEIGHT - 4
                        shape_y = mound_height

                fall_time = now

            if check_collision(current_shape, shape_x, shape_y + 1):
                for row in range(len(current_shape)):
                    for col in range(len(current_shape[row])):
                        if current_shape[row][col] == 1:
                            grid[shape_y + row][shape_x + col] = shape_color
                clear_rows()

                if check_perfect_clear(grid):
                    execute_perfect_clear(grid)

                if check_t_spin(grid, current_shape, shape_x, shape_y):
                    execute_t_spin(grid, current_shape, shape_x, shape_y)

                if any(grid[0]):
                    running = False

                shape_x, shape_y = GRID_WIDTH // 2 - len(current_shape[0]) // 2, 0
                current_shape_name = random.choice(list(SHAPES.keys()))
                current_shape = SHAPES[current_shape_name]["shape"]
                shape_color = SHAPES[current_shape_name]["color"]

            # Display the game state
            screen.fill(BLACK)
            draw_grid()
            draw_shape(current_shape, shape_x * GRID_SIZE, shape_y * GRID_SIZE, shape_color)

            for row in range(GRID_HEIGHT):
                for col in range(GRID_WIDTH):
                    if grid[row][col] != 0:
                        pygame.draw.rect(screen, grid[row][col], (col * GRID_SIZE, row * GRID_SIZE, GRID_SIZE, GRID_SIZE))

            pygame.display.flip()

        if pygame.time.get_ticks() - fall_time > screenshot_counter * screenshot_interval:
            screenshot_path = os.path.join(screenshot_dir, f"gen_{generation}_agent_{agent['fitness']}_screenshot_{screenshot_counter}.png")
            pygame.image.save(screen, screenshot_path)
            screenshot_counter += 1

        agent["fitness"] = heuristic_model(grid, current_shape, shape_x, shape_y)
        fitness_values = [agent['fitness'] for agent in population]
        average_fitness = np.mean(fitness_values)
        diversity = np.std(fitness_values)

        # Keep track of the best agent within the current generation
        if agent["fitness"] > best_agent_generation_fitness:
            best_agent_generation_fitness = agent["fitness"]

        reset_game()

    # Print or store the metrics
    print(f"Average Fitness: {average_fitness}, Diversity: {diversity}")
    print(f"Best agent's fitness in Generation {generation + 1}: {best_agent_generation_fitness}")

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1["genes"]) - 1)
    child_genes = np.concatenate((parent1["genes"][:crossover_point], parent2["genes"][crossover_point:]))
    return {"genes": child_genes, "fitness": 0}

def mutate(agent):
    mutation_rate = 0.8
    mutation_mask = np.random.choice([True, False], size=len(agent["genes"]), p=[mutation_rate, 1 - mutation_rate])
    agent["genes"][mutation_mask] = np.random.rand(np.sum(mutation_mask))
    return agent

def reset_game():
    global grid, score, last_lines_cleared
    grid = [[0] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
    score = 0
    last_lines_cleared = 0
    

# Initialize agents with random strategies
population = [{"genes": np.random.rand(4), "fitness": 0} for _ in range(POPULATION_SIZE)]

# Main loop
best_fitness_history = []  # Track the best fitness over generations

for generation in range(GENERATIONS):
    print(f"Generation {generation + 1}")
    selected_population = rank_based_selection(population)
    # Evaluate the fitness of each agent in the population
    for agent in population:
        evaluate_agent(agent, generation)

    # Sort the population based on fitness
    population.sort(key=lambda x: x["fitness"], reverse=True)

    # Print the best agent's fitness for this generation
    best_fitness = population[0]['fitness']
    print(f"Best agent's fitness in Generation {generation + 1}: {best_fitness}")

    best_agent = max(selected_population, key=lambda x: x['fitness'])
    best_fitness_history.append(best_agent['fitness'])

    # Check if the best agent can clear at least 3 lines at generation 10
    if generation == 9 and best_agent['fitness'] < 300:  # Adjust the threshold as needed
        # Modify the best agent's strategy or apply additional training
        print("Enhancing strategy for better line clearing...")
        # Your enhancement logic here...

    # Print the best agent's fitness and rank in the current generation
    print(f"Generation {generation}: Best fitness {best_agent['fitness']}, Rank {next(i+1 for i, ind in enumerate(selected_population) if ind is best_agent)}")

    # Save the best agent from the previous generation
    best_agent_previous_gen = copy.deepcopy(population[0])

    # Create a new population through crossover and mutation
    new_population = [best_agent_previous_gen]  # Keep the best agent from the previous generation

    for i in range(1, POPULATION_SIZE - 1, 2):  # Decrease the range to avoid going out of range
        if i < len(population) - 1:
            parent1 = population[i]
            parent2 = population[i + 1]
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])

    # Replace the old population with the new one
    population = new_population

# The best agent is the one with the highest fitness in the final generation
best_agent = population[0]

# Run the game with the best agent's strategy
evaluate_agent(best_agent, GENERATIONS)
pygame.quit()

# Plot the best fitness over generations
plt.plot(best_fitness_history)
plt.title('Best Fitness Over Generations')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()

replay_best_agent(best_agent)
