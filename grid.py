import pygame
import sys

# Initialize Pygame
pygame.init()

# Set the dimensions of the window
WINDOW_SIZE = (600, 600)
CELL_SIZE = WINDOW_SIZE[0] // 20  # Each cell will be a square of equal size
GRID_SIZE = 20

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Create the window
window = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Grid Example")

# Define the grid matrix
grid_matrix = [
    [0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)
]
grid_matrix2 = [
    [0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)
]

# Set some cells to 1 for demonstration
grid_matrix[5][5] = 5
grid_matrix[5][6] = 5
grid_matrix[5][7] = 5
grid_matrix[5][8] = 5
grid_matrix[5][9] = 5
grid_matrix[5][10] = 1
grid_matrix[11][5] = 1
grid_matrix[11][6] = 1
grid_matrix[11][7] = 1
grid_matrix[11][8] = 1
grid_matrix[11][9] = 1
grid_matrix[11][10] = 1


# Function to draw the grid
def draw_grid():
    for x in range(0, WINDOW_SIZE[0], CELL_SIZE):
        pygame.draw.line(window, BLACK, (x, 0), (x, WINDOW_SIZE[1]))
    for y in range(0, WINDOW_SIZE[1], CELL_SIZE):
        pygame.draw.line(window, BLACK, (0, y), (WINDOW_SIZE[0], y))

# Function to draw cells with value 1 in red and show the value in text
def draw_red_cells():
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if grid_matrix[row][col] > 0:
                rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                value = grid_matrix[row][col]
                color_intensity = int((value / 10) * 255)
                color = (color_intensity, 0, 0) 
                pygame.draw.rect(window, color, rect)
                font = pygame.font.Font(None, 10)
                text = font.render(str(grid_matrix[row][col]), True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                window.blit(text, text_rect)
            else:
                rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                font = pygame.font.Font(None, 10)
                text = font.render(str(grid_matrix[row][col]), True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                window.blit(text, text_rect)
            
def update_grid():
    # make the grid_matrix2 each cell value average of the surrounding cells
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            total = 0
            count = 0
            for i in range(-1, 1):
                for j in range(-1, 1):
                    if 0 <= row + i < GRID_SIZE and 0 <= col + j < GRID_SIZE:
                        total += grid_matrix[row + i][col + j]
                        count += 1
            grid_matrix2[row][col] = total / count
    # now make the grid_matrix equal to grid_matrix2
    #if it is the minimum value in sorrouding dont update it 

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            update = False ; 
            for i in range(-1, 1):
                for j in range(-1, 1):
                    if 0 <= row + i < GRID_SIZE and 0 <= col + j < GRID_SIZE:
                        if grid_matrix[row][col ]>= grid_matrix[row + i][col + j]:
                            update = True ; 
            if update == True : 
              grid_matrix[row][col] = grid_matrix2[row][col]
    # Print the grid_matrix2
    for row in grid_matrix2:
        print(row)

    
# Main loop
def main():
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n:
                    update_grid(); 

        window.fill(WHITE)  # Fill the window with white
        draw_grid()  # Draw the grid
        draw_red_cells()  # Draw red cells and text
        pygame.display.flip()  # Update the display

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
