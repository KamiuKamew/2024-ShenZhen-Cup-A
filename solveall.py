from itertools import combinations, product
from solveeq import get_solutions
def solveall(data):
    '''
    solve all combinations of the given data 
        (include position and time)
    :param data: [(x,y,z,t)s]
    :return: [(x,y,z,t)s]
    '''
    
    combination = list(combinations(data, 4))
    # combination = [((posi, time)*4)*many]

    result = []
    multi_result_num = 0
    for cmb in combination:
        solutions = get_solutions(cmb, 340)
        if len(solutions) > 1:
            multi_result_num += 1
        for sol in solutions:
            result.append(sol)
    #result = [(x,y,z,t)*many]

    return result, multi_result_num

if __name__ == '__main__':
    from origindata import origin_data
    from convertcoord import convert
    
    converted_datas = convert([(x,y,z,t) for x,y,z,ts in origin_data for t in ts])
    first_data = list(combinations(converted_datas, 4))[4]
    #result, multi_result_num = solveall(converted_datas)
    first_solutions = get_solutions(first_data, 340)

    #print(result)
    #print(multi_result_num) 
    print(first_solutions)
    from solveeq import plot_solutions
    plot_solutions(first_solutions)