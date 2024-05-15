import os
import hashlib

game_grid_path = "/Users/jianzhangli/src/BrickGame/full_grid"


def read_game(path):
    array = []
    with open(path, "r") as file:
        for line in file:
            row = [int(x) for x in line.strip().split(",")]
            array.append(row)
    return array


def write_game(path,array):
    with open(path, "w") as file:
        for row in array:
            file.write(",".join(str(x) for x in row) + "\n")


if __name__ == '__main__':
    for root, dirs, files in os.walk(game_grid_path):
        for file in files:
            file_path = os.path.join(root, file)
            print("File:", str(read_game(file_path)))

    print('main')

