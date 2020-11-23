import requests
from copy import deepcopy


def to_pddl_element(header, elements):
    output = '({}\n{})\n'

    body = ''
    for el in elements:
        body += el

    return output.format(header, body)


def good_cell(game_matrix, x, y):
    if x < 0 or x > len(game_matrix[0]) - 1 or y < 0 or y > len(game_matrix) - 1 or game_matrix[y][x] == 1:
        return False
    return True


def generate_roadmap(output, game_matrix, x, y):
    directions = [
        [x, y - 1],
        [x + 1, y],
        [x, y + 1],
        [x - 1, y]
    ]

    game_matrix[y][x] = -1

    for i in range(4):
        if good_cell(game_matrix, directions[i][0], directions[i][1]):
            output += '({} f_{}_{} f_{}_{})\n'.format(
                ('vertical' if i % 2 == 0 else 'horizontal'),
                str(x), str(y),
                str(directions[i][0]), str(directions[i][1])
            )
            if game_matrix[directions[i][1]][directions[i][0]] != -1:
                output = generate_roadmap(output, game_matrix, directions[i][0], directions[i][1])
    return output

def get_roadmap(game_matrix):
    roadmap = ''
    game_matrix = deepcopy(game_matrix)
    for y in range(len(game_matrix)):
        for x in range(len(game_matrix[y])):
            if game_matrix[y][x] == 0:
                roadmap = generate_roadmap('', game_matrix, x, y)
                break
    return roadmap


def generate_elements(game_matrix):
    inits, goals, objects = [], [], []
    for y in range(len(game_matrix)):
        for x in range(len(game_matrix[y])):
            if game_matrix[y][x] == 1:
                continue
            else:
                obj = 'f_{}_{} '.format(x, y)
                if game_matrix[y][x] == 0:
                    inits.append('(player {})\n'.format(obj))
                elif game_matrix[y][x] == 2:
                    goals.append('(box {})\n'.format(obj))
                elif game_matrix[y][x] == 3:
                    inits.append('(box {})\n'.format(obj))
                objects.append(obj)
    return inits, goals, objects



def gen_goal(goals):
    el_and = to_pddl_element('and', goals)
    goal = to_pddl_element(':goal', [el_and])
    return goal


class PDDLer:
    def __init__(self, domain_path, problem_path, output_path):
        self.domain = domain_path
        self.problem = problem_path
        self.output = output_path

    def generate_problem(self, game_matrix):
        roadmap = get_roadmap(game_matrix)
        inits, goals, objects = generate_elements(game_matrix)
        inits.append(roadmap)

        el_inits = to_pddl_element(':init', inits)
        el_goals = gen_goal(goals)
        el_objects = to_pddl_element(':objects', objects)
        el_domain = '(:domain sokoban)\n'

        problem_body = to_pddl_element('define (problem image_problem)', [el_domain, el_objects, el_inits, el_goals])

        with open(self.problem, 'w+') as f:
            f.write(problem_body)

    def solve(self):
        data = {'domain': open(self.domain, 'r').read(),
                'problem': open(self.problem, 'r').read()}

        resp = requests.post('http://solver.planning.domains/solve',
                             verify=False, json=data).json()
        print(resp)
        with open(self.output, 'w+') as f:
            f.write('\n'.join([act['name'] for act in resp['result']['plan']]))


# directory = 'C:/Users/Den/PycharmProjects/sokoban-image-processing/pddl'
# test = PDDLer(directory + '/domain.pddl', directory + '/problem.pddl', directory + '/plan.txt')
# game = [
#     [4, 0, 1],
#     [4, 3, 2],
#     [1, 1, 1]
# ]
# test.generate_problem(game)
# test.solve()
