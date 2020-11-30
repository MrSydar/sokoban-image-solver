from processors.image_game_processor import *
from processors.pddl import PDDLer
from processors.recognizer import Classifier


def process_image(path):
    image = get_image(path)
    thresh_with_symbols = custom_threshold(image)

    tmp = filter_play_field(thresh_with_symbols)
    thresh_grid = tmp[0]
    thresh_with_symbols = tmp[1]

    cells = get_cells(thresh_grid, 0.80)

    ordered_cells = order_cells(
        cells,
        sqrt(mean([cv2.contourArea(c) for c in cells])) / 2
    )

    return [thresh_with_symbols, ordered_cells]


def get_game_field(image, cells_contours):
    # TODO: easy to make not only NxN game. Barbeque data structure can return grid height and width
    N = sqrt(len(cells_contours))
    if N % 1 > 0:
        print("Error: Script in current stage can work only with NxN games. "
              "Check if all cells were recognized. N recognized: {}".format(N))
        exit(1)
    N = int(N)

    image_processor = Classifier(
        "C:/Users/Den/PycharmProjects/sokoban-image-processing/tfmodels/training_2/cp-0000.ckpt"
    )

    game_field = []

    for row in range(N):
        tmp = []
        for el in range(N):
            img = get_sign(image, cells_contours[row * N + el])
            if np.mean(img) >= 5:
                tmp.append(
                    image_processor.recognise(img)
                )
            else:
                tmp.append(4)
        game_field.append(tmp)

    return game_field


def solve(game, directory):
    solver = PDDLer(directory + '/domain.pddl', directory + '/problem.pddl', directory + '/output/plan.txt')
    solver.generate_problem(game)
    solver.solve()


def main():
    ret = process_image('images/samples/char-packs/squares_filled_3.jpg')
    thresh = ret[0]
    ordered_cells = ret[1]

    cv2.imshow('preview', thresh)
    cv2.waitKey()
    # game_field = get_game_field(thresh, ordered_cells)
    #
    # for row in game_field:
    #     print(row)
    #
    # solve(game_field, 'C:/Users/Den/PycharmProjects/sokoban-image-processing/pddl')


if __name__ == "__main__":
    main()
