from scripts.role_assignment import RoleAssigner
from scripts.llm_interface import GPTRolePrompter, SAMPLE_GPT_OUTPUT

from overcooked_role_assignment.common.subtasks import Subtasks

def calculate_roles(layout, task_list):
    """
    param layout: str, name of layout
    param task_list: list of str, list of subtasks in human readable format
    return: list of str, list of roles
    """

    role_generation = GPTRolePrompter()
    role_assigner = RoleAssigner(layout, None)

    # role_divisions = role_generation.process_gpt_response(SAMPLE_GPT_OUTPUT)
    # role_divisions = SAMPLE_GPT_OUTPUT
    role_divisions = role_generation.query_for_role_divisions(Subtasks.HUMAN_READABLE_ST)
    print(role_divisions)
    all_roles = {}
    for div in role_divisions:
        for role, subtasks in role_divisions[div].items():
            try:
                all_roles[role] = [Subtasks.IDS_TO_SUBTASKS[Subtasks.HR_SUBTASKS_TO_IDS[task]] for task in subtasks]
            except KeyError:
                pass
    # import pdb; pdb.set_trace()
    cost_dicts = role_assigner.evaluate_roles(all_roles)
    assignment = role_assigner.assign_roles(all_roles, cost_dicts, task_list)
    print(assignment)
    

if __name__ == "__main__":
    calculate_roles("ORA_symmetry", Subtasks.SUBTASKS[:-1])