# sokoban-image-solver

## Description
Solve your sokoban problem right from the image! Draw your problem with pen using special symbols, take a photo, 
load it into this solver and enjoy the solution. Awesome 🎉

## Requirements
You will need [tensorflow](https://www.tensorflow.org/install/pip) and [opencv](https://pypi.org/project/opencv-python/) installed.

## Prepare data
Draw a map using special characters:
`stickman` for player
`cross` for wall
`filled square` for box
`empty square` for box target position

Take a photo of your map and save it as in the JPG format.

### Example:
![game_1](https://user-images.githubusercontent.com/50991602/171459736-65376b00-f939-4de0-934f-bdaf83d5d1f9.jpg)

You can look into other examples under `./images/samples/games` directory.

## How to use
1. Run the tool with the next parameters:
- `i` : path to the image with problem 
```
python3 main.py -i ./images/samples/games/game_3.jpg
```

2. Wait until the end
3. Get solution from the `./pddl/output/plan.txt` file
