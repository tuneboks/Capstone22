# import tkinter as tk
#
# # Set number of rows and columns
# ROWS = 20
# COLS = 20
#
# # Create a grid of None to store the references to the tiles
# tiles = [[0 for _ in range(COLS)] for _ in range(ROWS)]
#
# def callback(event):
#     # Get rectangle diameters
#     col_width = c.winfo_width()//COLS
#     row_height = c.winfo_height()//ROWS
#     # Calculate column and row number
#     print(col_width, row_height)
#     col = event.x//col_width
#     row = event.y//row_height
#
#     # If the tile is not filled, create a rectangle
#     if not tiles[row][col]:
#         tiles[row][col] = c.create_rectangle(col*col_width, row*row_height, (col+1)*col_width, (row+1)*row_height, fill="red")
#
#
# # Create the window, a canvas and the mouse click event binding
# root = tk.Tk()
# w = 800
# h = 800
# c = tk.Canvas(root, width=w, height=h, borderwidth=5, background='white')
#
# for num in range(ROWS):
#     c.create_line(num, num * w/ROWS, num + w, num * w/ROWS, fill="black")
# for num in range(COLS):
#     c.create_line(num * w/ROWS, num, num * w/ROWS, num + w, fill="black")
#
# c.pack()
# c.bind("<B1-Motion>", callback)
# c.bind("<Button-1>", callback)
#
# root.mainloop()



# Import libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

size = 5
# # Create axis
axes = [size, size, size]
#
# # Create Data
# data = np.ones(axes, dtype=int)
#
# print(data)
data = np.random.choice(2, size=axes, p=[0.7, 0.3])



start = None

while start == None:
    x = np.random.randint(size)
    y = np.random.randint(size)
    z = np.random.randint(size)
    print(data[x][y][z])
    if data[x][y][z] == 0:
        data[x][y][z] = 2
    start = True

print(data)
# Control Transparency
alpha = 0.9

# # Control colour
# colors = np.empty(axes + [4], dtype=np.float32)
#
# colors[0] = [1, 0, 0, alpha]  # red
# colors[1] = [0, 1, 0, alpha]  # green
# colors[2] = [0, 0, 1, alpha]  # blue
# colors[3] = [1, 1, 0, alpha]  # yellow
# colors[4] = [1, 1, 1, alpha]  # grey

# Plot figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Voxels is used to customizations of
# the sizes, positions and colors.
ax.voxels(data, edgecolors='black')

plt.show()
