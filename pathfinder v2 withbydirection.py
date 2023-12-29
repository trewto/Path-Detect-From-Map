import pygame
import sys
import random
from math import sqrt
import cv2
import numpy as np
from PIL import Image
import copy
# Constants
WIDTH, HEIGHT = 630, 630
ROWS, COLS = 210,210
CELL_WIDTH = WIDTH // COLS
CELL_HEIGHT = HEIGHT // ROWS
 
# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
RED = (255,0,0)
YELLOW= (255,255,0)

# Create grid and cellBlocked matrix
grid = [[{'x': row, 'y': col, 'is_blocked': False, 'is_visited': False, 'distance': float('inf') , 'previous' : None}
         for col in range(COLS)] for row in range(ROWS)]


#cellBlocked = [[False for _ in range(COLS)] for _ in range(ROWS)]  # Initialize all cells as unblocked
#grid[2][3]['is_blocked'] = True

def display_multiline_text(window, text, position, font_size=21, color=BLACK):
    font = pygame.font.Font(None, font_size)
    lines = text.split("|")  # Split text into lines based on a separator character "|"

    y_offset = 0
    for line in lines:
        text_surface = font.render(line, True, color)
        window.blit(text_surface, (position[0], position[1] + y_offset))
        y_offset += font_size  # Adjust the Y offset for each line

#print(grid)
# Pygame setup
pygame.init()
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Grid")
font = pygame.font.SysFont(None, 20)


def add_random_blocks(probability):
    for row in range(ROWS):
        for col in range(COLS):
            if random.random() < probability:  # Using probability to randomly block cells
                grid[row][col]['is_blocked'] = True
#add_random_blocks(0.05)
def draw_grid():
    window.fill(WHITE)
    for row in range(ROWS):
        for col in range(COLS):
            if grid[row][col]['is_blocked']:
                pygame.draw.rect(window, RED, (col * CELL_WIDTH, row * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))
            else:
                pygame.draw.rect(window, GREY, (col * CELL_WIDTH, row * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT), 1)
            
            if grid[row][col]['is_visited']:
                pygame.draw.rect(window, YELLOW, (col * CELL_WIDTH, row * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT), 1)
            if grid2[row][col]['is_visited']:
                pygame.draw.rect(window, YELLOW, (col * CELL_WIDTH, row * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT), 1)

            cell_info = f'({grid[row][col]["x"]}, {grid[row][col]["y"]}) | V: {grid[row][col]["is_visited"]} | D: {grid[row][col]["distance"]}'
            #display_multiline_text(window, cell_info, (col * CELL_WIDTH + 5, row * CELL_HEIGHT + 5))  # Adjust position for better alignment
   
    #pygame.display.update()

def draw_cell(x,y,color=(0,255,22)):
    pygame.draw.rect(window, color, (y * CELL_WIDTH, x * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT))

running = True

start = [10,10]

start = [ROWS-1,COLS-1]
start = [ROWS-1,int(COLS/2)]
end = [ROWS-1,COLS-1]


end = [2,2]
end = [0,int(COLS/2)]
#please uncomment this to run properly without mouse clcik
#grid[start[0]][start[1]]['distance']=0
 
grid[start[0]][start[1]]['is_blocked'] = False
grid[end[0]][end[1]]['is_blocked'] = False

index = start






def convert_img(image_path = "United3.png"):
    # Replace 'United3.png' with the path to your image file
    

    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Check if the image is loaded successfully
    if image is not None:
        #cv2.imshow('Image', image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # Define the road color range#250,236,192#rgba(234,203,138,255)#250,238,192
        lower_road_color = np.array([0,0, 240])
        upper_road_color = np.array([255, 255, 255])

        # Mask the image to extract road pixels
        mask = cv2.inRange(image, lower_road_color, upper_road_color)
        result = cv2.bitwise_and(image, image, mask=mask)
        grayscale_image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Create a kernel for dilation
        kernel = np.ones((5, 5), np.uint8)

        # Perform dilation on the black and white image
        dilated_image = cv2.dilate(grayscale_image, kernel, iterations=1)

        cv2.imwrite("dilated_image.png", dilated_image)
        return "dilated_image.png"
        #return dilated_image
    else:
        print("Failed to load image")  # Add this line to indicate failure in loading the image
        return None




#img part start
#image_path = "Untitled5.png"
initial_img = "United3.png"
initial_img = "United3.png"
image_path = convert_img(initial_img)   

image = pygame.image.load(image_path)


img_width, img_height = image.get_size()


# Resize the image to match grid dimensions
resized_image = pygame.transform.scale(image, (COLS, ROWS))
resized_image2 = pygame.transform.scale(image, (COLS*CELL_WIDTH, ROWS*CELL_HEIGHT))

image1 = pygame.image.load(initial_img)
resized_image1 = pygame.transform.scale(image1, (COLS*CELL_WIDTH, ROWS*CELL_HEIGHT))


color_lower_range = (0, 0, 0)  # Example lower range (R, G, B)
color_upper_range = (  10, 10, 10)  # Example upper range (R, G, B)

for row in range(ROWS):
    for col in range(COLS):
        pixel_color = resized_image.get_at((col, row))  # Get pixel color at (x, y)
        # Check if the pixel color is within the defined range
        if (color_lower_range[0] <= pixel_color[0] <= color_upper_range[0] and
            color_lower_range[1] <= pixel_color[1] <= color_upper_range[1] and
            color_lower_range[2] <= pixel_color[2] <= color_upper_range[2]):
            grid[row][col]['is_blocked'] = True
#img part end


  





def update_neghibour_distance(x,y,index_distance):

    grid[x][y]['is_visited'] = True 
    if x > 0 and not grid[x - 1][y]['is_blocked'] and not grid[x - 1][y]['is_visited']:
        if index_distance + 1 <grid[x - 1][y]['distance']:
            grid[x - 1][y]['distance'] =index_distance + 1 
            grid[x - 1][y]['previous'] = (x,y)
    if x < ROWS - 1 and not grid[x + 1][y]['is_blocked'] and not grid[x + 1][y]['is_visited']:
        if index_distance + 1 <grid[x + 1][y]['distance']:
            grid[x + 1][y]['distance'] =index_distance + 1 
            grid[x + 1][y]['previous'] = (x,y)
    if y > 0 and not grid[x][y - 1]['is_blocked'] and not grid[x][y - 1]['is_visited']:
         if index_distance + 1 < grid[x][y - 1]['distance']:
            grid[x][y - 1]['distance']  =index_distance + 1 
            grid[x][y - 1]['previous'] = (x,y)
    if y < COLS - 1 and not grid[x][y + 1]['is_blocked'] and not grid[x][y + 1]['is_visited']:
         if index_distance + 1 <grid[x][y + 1]['distance']:
            grid[x][y + 1]['distance'] =index_distance + 1 
            grid[x][y + 1]['previous'] = (x,y)


def update_neghibour_distance2(x,y,index_distance):
    grid2[x][y]['is_visited'] = True 
    #grid[x][y]['is_visited'] = True 
    if x > 0 and not grid2[x - 1][y]['is_blocked'] and not grid2[x - 1][y]['is_visited']:
        if index_distance + 1 <grid2[x - 1][y]['distance']:
            grid2[x - 1][y]['distance'] =index_distance + 1 
            grid2[x - 1][y]['previous'] = (x,y)
    if x < ROWS - 1 and not grid2[x + 1][y]['is_blocked'] and not grid2[x + 1][y]['is_visited']:
        if index_distance + 1 <grid2[x + 1][y]['distance']:
            grid2[x + 1][y]['distance'] =index_distance + 1 
            grid2[x + 1][y]['previous'] = (x,y)
    if y > 0 and not grid2[x][y - 1]['is_blocked'] and not grid2[x][y - 1]['is_visited']:
         if index_distance + 1 < grid2[x][y - 1]['distance']:
            grid2[x][y - 1]['distance']  =index_distance + 1 
            grid2[x][y - 1]['previous'] = (x,y)
    if y < COLS - 1 and not grid2[x][y + 1]['is_blocked'] and not grid2[x][y + 1]['is_visited']:
         if index_distance + 1 <grid2[x][y + 1]['distance']:
            grid2[x][y + 1]['distance'] =index_distance + 1 
            grid2[x][y + 1]['previous'] = (x,y)

# Your existing code...
def backline(last,color=(0, 255, 0)):
    if grid[last[0]][last[1]]['is_visited']:
        path = []
        current = last
        while current:
            path.append(current)
            current = grid[current[0]][current[1]]['previous']

        # Drawing the shortest path
        for node in path:
            draw_cell(node[0], node[1], color)  # Green color for the shortest path
        pygame.display.update()

def backline2(last,color=(0, 255, 0)):
    if grid2[last[0]][last[1]]['is_visited']:
        path = []
        current = last
        while current:
            path.append(current)
            current = grid2[current[0]][current[1]]['previous']

        # Drawing the shortest path
        for node in path:
            draw_cell(node[0], node[1], color)  # Green color for the shortest path
        pygame.display.update()


def distance(a, b):
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

skip = False 
reached = False
initial_selcted = False
end_selected = False
img_active= 1 #1 or 2 or 3
index2 = end
grid2 = copy.deepcopy(grid)
mid = None
while running:


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_i:
                img_active =  img_active+1
                if img_active>3:
                    img_active = 1

        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            row = mouse_y // CELL_HEIGHT
            col = mouse_x // CELL_WIDTH
            #print (pygame.mouse.get_pos())
            print((row,col))
        

            if  not  initial_selcted :
                #start[0] = row
                #start[1] = col
                start = (row,col)
                grid[start[0]][start[1]]['distance']=0
                initial_selcted = True
                index = start
                print (initial_selcted)
            elif not end_selected:
                end[0] = row
                end[1] = col
                end_selected = True
                print (end_selected)
                index2= end
                grid2[end[0]][end[1]]['distance']=0
                print (grid2)


            #if not (row == start[0] and col == start[1]) and not (row == end[0] and col == end[1]):
            #    grid[row][col]['is_blocked'] = True


    min_distance = float('inf')
    min_node = None

    min_distance2 = float('inf')
    min_node2 = None


    # Finding the unvisited node with the minimum distance
    ATTRACTION_FACTOR = 1.4
    ATTRACTION_FACTOR =5

    ATTRACTION_FACTOR = 4
    if not skip and initial_selcted==True and end_selected ==True:
        
        for row in range(ROWS):
            for col in range(COLS):
                if not grid[row][col]['is_visited'] and (grid[row][col]['distance'] + ATTRACTION_FACTOR* distance( (row, col) , (end[0],end[1]))) < min_distance:
                    min_distance = grid[row][col]['distance'] + ATTRACTION_FACTOR*  distance( (row, col) , (end[0],end[1]))
                    min_node = (row, col)
                   
                    if row == end[0] and col == end[1]:
                        #skip = True # if skip == true , when it reach the destination it will stop
                        #skip = false ; i want more try3
                        print("Reached")
                        reached= True
                
                if not grid2[row][col]['is_visited'] and (grid2[row][col]['distance'] + ATTRACTION_FACTOR* distance( (row, col) , (start[0],start[1]))) < min_distance2:
                    min_distance2 = grid2[row][col]['distance'] + ATTRACTION_FACTOR*  distance( (row, col) , (start[0],start[1]))
                    min_node2 = (row, col)
                    print(5)

                if grid[row][col]['is_visited'] and  grid2[row][col]['is_visited']:
                    skip = True
                    mid = (row,col)
                    print(99)
                    
                    
                    

        if min_node:
            index = min_node
            update_neghibour_distance(index[0],index[1],grid[index[0]][index[1]]['distance'])
            backline(min_node)
        if min_node2:
            index2 = min_node2
            update_neghibour_distance2(index2[0],index2[1],grid2[index2[0]][index2[1]]['distance'])
            backline2(min_node2)
        
            
    else:
        t=1
        backline(end)
        backline2(start)
             


    draw_grid()
    if img_active==2:
        window.blit(resized_image2, (0, 0))
    elif img_active==1:
        window.blit(resized_image1, (0, 0))
    draw_cell(start[0], start[1])
    draw_cell(end[0], end[1])
    backline((end[0],end[1]))
    backline2((start[0],start[1]))
    if mid is not None:
         backline((mid[0],mid[1]))
         backline2((mid[0],mid[1]))
    pygame.display.update()
    