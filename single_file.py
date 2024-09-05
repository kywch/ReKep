# Follow python main.py --use_cached_query --visualize
import os
import json
import time

import numpy as np

from main import Main
from utils import (
    bcolors,
    get_config,
    load_functions_from_txt,
    # get_linear_interpolation_steps,
    # spline_interpolate_poses,
    get_callable_grasping_cost_fn,
    # print_opt_debug_dict,
)

task_list = {
    'pen': {
        'scene_file': './configs/og_scene_file_pen.json',
        'instruction': 'reorient the white pen and drop it upright into the black pen holder',
        'rekep_program_dir': './vlm_query/pen',
        },
}

task = task_list['pen']
scene_file = task['scene_file']
instruction = task['instruction']

global_config = get_config(config_path="./configs/config.yaml")

start_time = time.time()
# Headless or not, it takes about 3 min to init ... 
# from omnigibson.macros import gm
# gm.HEADLESS = True
main = Main(scene_file, visualize=True)
print("Time to init Main: ", int(time.time() - start_time), "s")

#####################################################
# main.perform_task(instruction, rekep_program_dir=task['rekep_program_dir'])
main.env.reset()

### Get camera observation ###
cam_obs = main.env.get_cam_obs()

CAMERA_TO_USE_FOR_VLM = 0
rgb = cam_obs[CAMERA_TO_USE_FOR_VLM]['rgb']
points = cam_obs[CAMERA_TO_USE_FOR_VLM]['points']  # point coords (x, y, z) in the camera frame of shape (N, 3)
mask = cam_obs[CAMERA_TO_USE_FOR_VLM]['seg']

# import matplotlib.pyplot as plt
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
# ax1.imshow(rgb)
# ax2.imshow(points)
# ax3.imshow(mask)

### Keypoint proposal ###
keypoints, projected_img = main.keypoint_proposer.get_keypoints(rgb, points, mask)
print(f'{bcolors.HEADER}Got {len(keypoints)} proposed keypoints{bcolors.ENDC}')
main.visualizer.show_img(projected_img)

### Constraint generation from the keypoints ###
# from the image, instruction, and metadata, generate a directory called rekep_program_dir
# Use the provided vlm_query for now. Try this with claude or the open-source ones
# metadata = {'init_keypoint_positions': keypoints, 'num_keypoints': len(keypoints)}
# rekep_program_dir = self.constraint_generator.generate(projected_img, instruction, metadata)

# vlm proposes num_stages, and subgoal & path constraints fns for each stage
# For grasping and releasing, write the keypoint index. For non-grasping stage, write -1.

## The metadata contains grasp and release keypoints, so it'd be good to check the validity here.
# self.is_grasp_stage = self.program_info['grasp_keypoints'][self.stage - 1] != -1
# self.is_release_stage = self.program_info['release_keypoints'][self.stage - 1] != -1
# # can only be grasp stage or release stage or none
# assert self.is_grasp_stage + self.is_release_stage <= 1, "Cannot be both grasp and release stage"

#####################################################
# self._execute(rekep_program_dir, disturbance_seq=None)

rekep_program_dir = './vlm_query/pen'

with open(os.path.join(rekep_program_dir, 'metadata.json'), 'r') as f:
    main.program_info = json.load(f)

# register keypoints to be tracked
main.env.register_keypoints(main.program_info['init_keypoint_positions'])

# keypoints in the world frame of shape (N, 3)
for idx, keypoint in enumerate(main.env.get_keypoint_positions()):
    print(f"Keypoint {idx}: {keypoint}, {main.env.get_object_by_keypoint(idx).name}")

# load constraints
main.constraint_fns = dict()
for stage in range(1, main.program_info['num_stages'] + 1):  # stage starts with 1
    stage_dict = dict()
    for constraint_type in ['subgoal', 'path']:
        load_path = os.path.join(rekep_program_dir, f'stage{stage}_{constraint_type}_constraints.txt')
        get_grasping_cost_fn = get_callable_grasping_cost_fn(main.env)  # special grasping function for VLM to call
        stage_dict[constraint_type] = load_functions_from_txt(load_path, get_grasping_cost_fn) if os.path.exists(load_path) else []
    main.constraint_fns[stage] = stage_dict

for stage, stage_dict in main.constraint_fns.items():
    print(f"Stage {stage} constraints: {stage_dict}")

# bookkeeping of which keypoints can be moved in the optimization
main.keypoint_movable_mask = np.zeros(main.program_info['num_keypoints'] + 1, dtype=bool)
main.keypoint_movable_mask[0] = True  # first keypoint is always the ee, so it's movable

### main loop: going through the stages
main._update_stage(1)
while True:
    print(f"\nStage {main.stage}, updating coordinates...")
    # keypoints and the ee pose in the world frame
    scene_keypoints = main.env.get_keypoint_positions()
    main.keypoints = np.concatenate([[main.env.get_ee_pos()], scene_keypoints], axis=0)  # first keypoint is always the ee
    main.curr_ee_pose = main.env.get_ee_pose()
    main.curr_joint_pos = main.env.get_arm_joint_postions()

    # SDF (signed distance field) and collision points
    main.sdf_voxels = main.env.get_sdf_voxels(main.config['sdf_voxel_size'])
    main.collision_points = main.env.get_collision_points()

    # NOTE: In the provided example, the path constraints only check if the robot is still grasping the keypoint
    # ====================================
    # = decide whether to backtrack
    # constraint_tolerance: 0.10
    # ====================================
    backtrack = False
    if main.stage > 1:
        path_constraints = main.constraint_fns[main.stage]['path']
        # constraints are functions generated by the VLM, and loaded with exec_safe
        # "get_grasping_cost_by_keypoint_idx" fn is "get_grasping_cost_fn" which returns 0 if the grasping is valid and 1 if not
        
        for constraints in path_constraints:
            violation = constraints(main.keypoints[0], main.keypoints[1:])
            if violation > main.config['constraint_tolerance']:
                backtrack = True
                print(f"\nStage {main.stage} path violation!")
                break

    if backtrack:
        # determine which stage to backtrack to based on constraints
        for new_stage in range(main.stage - 1, 0, -1):
            path_constraints = main.constraint_fns[new_stage]['path']
            # if no constraints, we can safely backtrack
            if len(path_constraints) == 0:
                break
            # otherwise, check if all constraints are satisfied
            all_constraints_satisfied = True
            for constraints in path_constraints:
                violation = constraints(main.keypoints[0], main.keypoints[1:])
                if violation > main.config['constraint_tolerance']:
                    all_constraints_satisfied = False
                    break
            if all_constraints_satisfied:   
                break
        print(f"{bcolors.HEADER}[stage={main.stage}] backtrack to stage {new_stage}{bcolors.ENDC}")
        main._update_stage(new_stage)

    # This is the main execution loop for each stage
    else:
        # ====================================
        # = motion planning, using the "off-the-shelf" IK solver, i.e., IssacSim's Lula Kinematics Solver
        # https://docs.omniverse.nvidia.com/isaacsim/latest/advanced_tutorials/tutorial_motion_generation_lula_kinematics.html
        # ====================================

        # Using main.subgoal_solver to get subgoal pose, which minimizes the subgoal constraints fn
        print(f"\nStage {main.stage}, solving next subgoal...")
        next_subgoal = main._get_next_subgoal(from_scratch=main.first_iter)

        # main.visualizer.visualize_subgoal(next_subgoal)

        # using main.path_solver to get path + gripper action (0/1)
        print(f"\nStage {main.stage}, solving next path toward subgoal...")
        next_path = main._get_next_path(next_subgoal, from_scratch=main.first_iter)
        # main.visualizer.visualize_path(next_path)

        main.first_iter = False
        main.action_queue = next_path.tolist()
        print(f"\nStage {main.stage}, {len(main.action_queue)} actions in the queue...")

        # ====================================
        # = execute the action queue
        # action_steps_per_iter: 5 (default) -- 5 steps per iteration
        # after each iteration, repeat the motion planning from the current pose
        # ====================================
        count = 0
        while len(main.action_queue) > 0 and count < main.config['action_steps_per_iter']:
            next_action = main.action_queue.pop(0)
            # precise: whether to use small position and rotation thresholds for precise movement (robot would move slower)
            precise = len(main.action_queue) == 0
            main.env.execute_action(next_action, precise=precise)
            count += 1
            print(f"\nStage {main.stage}, {count} out of {count + len(main.action_queue)} actions executed...")

        if len(main.action_queue) == 0:
            if main.is_grasp_stage:
                print(f"\nStage {main.stage}, executing grasp action...")
                main._execute_grasp_action()
            elif main.is_release_stage:
                print(f"\nStage {main.stage}, executing release action...")
                main._execute_release_action()

            # if completed, save video and return
            if main.stage == main.program_info['num_stages']: 
                print(f"\nTask completed, saving video...")
                main.env.sleep(2.0)
                # save_video writes all the video_cache (not just from the current stage) to a file
                save_path = main.env.save_video()
                print(f"{bcolors.OKGREEN}Video saved to {save_path}\n\n{bcolors.ENDC}")
                exit()

            else:
                # progress to next stage
                print(f"\nStage {main.stage} done...")
                main._update_stage(main.stage + 1)
