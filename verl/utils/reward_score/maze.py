import re
from . import Verifier

def extract_answer(solution_str):
    #solution = [sol.split("<answer>")[-1].split("<answer>").strip() for sol in solution_str]
    #solution = solution_str[0][0].split("<answer>")[-1].split("<answer>").strip()
    solution = re.search("<answer>[A-Z]+", solution_str)

    if solution is None:
        final_answer = None
    else:
        final_answer = solution.group(0)
        final_answer = final_answer.replace("<answer>", "").replace(" ", "")

    return final_answer

def compute_score(solution_str, current_map, format_score=0.05, score=1.):
    current_map = current_map.replace("\n", "")
    if(len(current_map) != 16):
        return 0
    reward = 0
    pattern = r"\s*<answer>.*?</answer>"
    match = re.match(pattern, solution_str)
    if not match:
        reward = reward - format_score
    
    reward = reward - 0.001 * abs(len(solution_str) - 512)

    answer = extract_answer(solution_str)
    if answer is None:
        return reward
    elif Verifier.VerifierMaze(current_map).verify(answer):
          reward = reward + score

    return reward
