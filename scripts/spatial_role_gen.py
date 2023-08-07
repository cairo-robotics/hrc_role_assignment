from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

def generate_spatial_roles_simple(mdp, mlam, subtasks):
    jobs_by_area = {
        "top" : [],
        "bottom" : [],
        "left" : [],
        "right" : []
    }
    start_fn = mdp.get_subtask_start_state_fn(mlam)
    for subtask in subtasks:
        start_state = start_fn(subtask)
        x, y = start_state.players[0].position
        if x < mdp.shape[0] / 2:
            jobs_by_area["left"].append(subtask)
        else:
            jobs_by_area["right"].append(subtask)

        if y < mdp.shape[1] / 2:
            jobs_by_area["bottom"].append(subtask)
        else:
            jobs_by_area["top"].append(subtask)

    return jobs_by_area
