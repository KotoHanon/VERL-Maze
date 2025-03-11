import re

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

def compute_score(solution_str, ground_truth, format_score=0.2, score=1.):
    reward = 0
    pattern = r"\s*<answer>.*?</answer>"
    match = re.match(pattern, solution_str)
    if match:
        reward = reward + format_score

    answer = extract_answer(solution_str)
    if answer == ground_truth:
          reward = reward + score

    return reward
