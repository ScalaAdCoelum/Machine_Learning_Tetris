import pygame
import random
import numpy as np

# Constants
WIDTH, HEIGHT = 300, 600
GRID_SIZE = 30
GRID_WIDTH, GRID_HEIGHT = WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE
WHITE, BLACK = (255, 255, 255), (0, 0, 0)

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

# Genetic Algorithm Constants
POPULATION_SIZE = 100   
GENERATIONS = 500

# Functions for Genetic Algorithm
def evaluate_agent(agent):
    global grid, score, last_lines_cleared, fall_speed

    # Reset the game state
    reset_game()

    fall_speed = 500  # Add this line to define fall_speed

    for generation in range(GENERATIONS):
        current_shape_name = random.choice(list(SHAPES.keys()))
        current_shape = SHAPES[current_shape_name]["shape"]
        shape_color = SHAPES[current_shape_name]["color"]

        shape_x, shape_y = GRID_WIDTH // 2 - len(current_shape[0]) // 2, 0
        fall_time = 0
        running = True

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

                fall_time = now

            if check_collision(current_shape, shape_x, shape_y + 1):
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

            # Display the game state
            screen.fill(BLACK)
            draw_grid()
            draw_shape(current_shape, shape_x * GRID_SIZE, shape_y * GRID_SIZE, shape_color)

            for row in range(GRID_HEIGHT):
                for col in range(GRID_WIDTH):
                    if grid[row][col] != 0:
                        pygame.draw.rect(screen, grid[row][col], (col * GRID_SIZE, row * GRID_SIZE, GRID_SIZE, GRID_SIZE))

            pygame.display.flip()

        agent["fitness"] = score
        print(f"Generation {generation + 1}, Fitness: {agent['fitness']}")
        reset_game()


def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1["genes"]) - 1)
    child_genes = np.concatenate((parent1["genes"][:crossover_point], parent2["genes"][crossover_point:]))
    return {"genes": child_genes, "fitness": 0}

def mutate(agent):
    mutation_rate = 0.1
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
for generation in range(GENERATIONS):
    print(f"Generation {generation + 1}")

    # Evaluate the fitness of each agent in the population
    for agent in population:
        evaluate_agent(agent)

    # Sort the population based on fitness
    population.sort(key=lambda x: x["fitness"], reverse=True)

    # Print the best agent's fitness for this generation
    print(f"Best fitness: {population[0]['fitness']}")

    # Create a new population through crossover and mutation
    new_population = []
    for i in range(0, POPULATION_SIZE, 2):
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
print(f"Best agent's fitness: {best_agent['fitness']}")

# Run the game with the best agent's strategy
evaluate_agent(best_agent)
pygame.quit()
