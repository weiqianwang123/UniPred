"""Main entry point for running approaches in environments.

Example usage with oracle NSRTs:
    python predicators/main.py --env satellites --approach oracle --seed 0

Example with verbose logging:
    python predicators/main.py --env satellites --approach oracle --seed 0 \
        --debug

Example usage with IVNTR (bilevel learning of neural predicates):
    python predicators/main.py --env satellites --approach ivntr --seed 0 \
        --excluded_predicates "ViewClear,IsCalibrated,HasChemX,HasChemY,Sees" \
        --neupi_pred_config predicators/config/satellites/pred.yaml

To test a saved approach instead of learning (for example on the
in-domain test tasks):
    python predicators/main.py --env satellites --approach ivntr --seed 0 \
        --load_approach --in_domain_test True

To load saved data:
    python predicators/main.py --env satellites --approach ivntr --seed 0 \
        --load_data

To make videos of test tasks:
    python predicators/main.py --env satellites --approach oracle --seed 0 \
        --make_test_videos --num_test_tasks 10

A learning approach learns from an offline dataset of demonstrations of the
train tasks (or loads what it learned before) and is then evaluated once on
the test tasks. Fast Downward is used by the default task planner: set
FD_EXEC_PATH to its directory (see planning.py). PYTHONHASHSEED must be 0.
"""

import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List

import dill as pkl
import torch.multiprocessing as mp

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout, \
    create_approach
from predicators.cogman import CogMan, run_episode_and_get_observations
from predicators.datasets import create_dataset
from predicators.envs import BaseEnv, create_new_env
from predicators.execution_monitoring import create_execution_monitor
from predicators.ground_truth_models import get_gt_options, \
    parse_config_included_options
from predicators.perception import create_perceiver
from predicators.settings import CFG
from predicators.structs import Metrics


def main() -> None:
    """Main entry point for running approaches in environments."""
    assert os.environ.get("PYTHONHASHSEED") == "0", \
        "Please add `export PYTHONHASHSEED=0` to your bash profile!"
    script_start = time.perf_counter()
    # Parse & validate args
    args = utils.parse_args()
    utils.update_config(args)
    str_args = " ".join(sys.argv)
    # Log to stderr.
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if CFG.log_file:
        handlers.append(logging.FileHandler(CFG.log_file, mode='w'))
    logging.basicConfig(level=CFG.loglevel,
                        format="%(message)s",
                        handlers=handlers,
                        force=True)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    if CFG.log_file:
        logging.info(f"Logging to {CFG.log_file}")
    logging.info(f"Running command: python {str_args}")
    logging.info("Full config:")
    logging.info(CFG)
    logging.info(f"Git commit hash: {utils.get_git_commit_hash()}")
    os.makedirs(CFG.results_dir, exist_ok=True)
    os.makedirs(CFG.eval_trajectories_dir, exist_ok=True)
    # Create classes. Note that seeding happens inside the env and approach.
    env = create_new_env(CFG.env, do_cache=True, use_gui=CFG.use_gui)
    # The action space needs to be seeded externally, because env.action_space
    # is often created during env __init__().
    env.action_space.seed(CFG.seed)
    assert env.goal_predicates.issubset(env.predicates)
    # The known predicates are passed into the approach and into dataset
    # creation.
    preds, _ = utils.parse_config_excluded_predicates(env)
    # We assume that a train Task can be constructed from an
    # EnvironmentTask, i.e., that the initial observation determines all of
    # the objects and their initial states: all data collection is offline.
    perceiver = create_perceiver(CFG.perceiver)
    train_tasks = [perceiver.reset(t) for t in env.get_train_tasks()]
    # If train tasks have goals that involve excluded predicates, strip those
    # predicate classifiers to prevent leaking information to the approaches.
    approach_train_tasks = [
        utils.strip_task(task, preds) for task in train_tasks
    ]
    if CFG.option_learner == "no_learning":
        # If we are not doing option learning, pass in all the environment's
        # oracle options.
        options = get_gt_options(env.get_name())
    else:
        # Determine from the config which oracle options to include, if any.
        options = parse_config_included_options(env)
    approach = create_approach(CFG.approach, preds, options, env.types,
                               env.action_space, approach_train_tasks)
    execution_monitor = create_execution_monitor(CFG.execution_monitor)
    cogman = CogMan(approach, perceiver, execution_monitor)
    learning_time = 0.0
    if approach.is_learning_based:
        if CFG.load_approach:
            cogman.load()
        else:
            # Demonstrations are generated with the non-stripped train tasks,
            # because the demonstrator may need the oracle predicates.
            dataset = create_dataset(env, train_tasks, options)
            logging.info("Using {} device for training".format(CFG.device))
            if 'cuda' in CFG.device:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            learning_start = time.perf_counter()
            cogman.learn_from_offline_dataset(dataset)
            learning_time = time.perf_counter() - learning_start
    offline_learning_metrics = {
        f"offline_learning_{k}": v
        for k, v in cogman.metrics.items()
    }
    results = _run_testing(env, cogman)
    results["learning_time"] = learning_time
    results.update(offline_learning_metrics)
    _save_test_results(results)
    script_time = time.perf_counter() - script_start
    logging.info(f"\n\nMain script terminated in {script_time:.5f} seconds")


def _run_testing(env: BaseEnv, cogman: CogMan) -> Metrics:
    logging.info("Running test tasks...")
    test_tasks = env.get_test_tasks()
    num_found_policy = 0
    num_solved = 0
    cogman.reset_metrics()
    total_suc_time = 0.0
    total_low_level_action_cost = 0.0
    total_num_solve_timeouts = 0
    total_num_solve_failures = 0
    total_num_execution_timeouts = 0
    total_num_execution_failures = 0

    save_prefix = utils.get_config_path_str()
    metrics: Metrics = defaultdict(float)
    curr_num_nodes_created = 0.0
    curr_num_nodes_expanded = 0.0
    for test_task_idx, env_task in enumerate(test_tasks):
        solve_start = time.perf_counter()
        try:
            # We call reset here, outside of run_episode_and_get_observations,
            # so that we can log planning failures, timeouts, etc.
            cogman.reset(env_task)
        except (ApproachTimeout, ApproachFailure) as e:
            logging.info(f"Task {test_task_idx+1} / {len(test_tasks)}: "
                         f"Approach failed to solve with error: {e}")
            if isinstance(e, ApproachTimeout):
                total_num_solve_timeouts += 1
            elif isinstance(e, ApproachFailure):
                total_num_solve_failures += 1
            if CFG.make_failure_videos and e.info.get("partial_refinements"):
                video = utils.create_video_from_partial_refinements(
                    e.info["partial_refinements"], env, "test", test_task_idx,
                    CFG.horizon)
                outfile = f"{save_prefix}__task{test_task_idx+1}_failure.mp4"
                utils.save_video(outfile, video)
            if CFG.crash_on_failure:
                raise e
            continue
        solve_time = time.perf_counter() - solve_start
        metrics[f"PER_TASK_task{test_task_idx}_solve_time"] = solve_time
        metrics[
            f"PER_TASK_task{test_task_idx}_nodes_created"] = cogman.metrics[
                "total_num_nodes_created"] - curr_num_nodes_created
        metrics[
            f"PER_TASK_task{test_task_idx}_nodes_expanded"] = cogman.metrics[
                "total_num_nodes_expanded"] - curr_num_nodes_expanded
        curr_num_nodes_created = cogman.metrics["total_num_nodes_created"]
        curr_num_nodes_expanded = cogman.metrics["total_num_nodes_expanded"]

        num_found_policy += 1
        make_video = False
        solved = False
        caught_exception = False
        if CFG.make_test_videos or CFG.make_failure_videos:
            monitor = utils.VideoMonitor(env.render)
        else:
            monitor = None
        try:
            # Now, measure success by running the policy in the environment.
            traj, solved, execution_metrics = run_episode_and_get_observations(
                cogman,
                env,
                "test",
                test_task_idx,
                max_num_steps=CFG.horizon,
                monitor=monitor)
            num_opt = execution_metrics["num_options_executed"]
            metrics[f"PER_TASK_task{test_task_idx}_options_executed"] = num_opt
            exec_time = execution_metrics["policy_call_time"]
            metrics[f"PER_TASK_task{test_task_idx}_exec_time"] = exec_time
            if CFG.refinement_data_include_execution_cost:
                total_low_level_action_cost += (
                    len(traj[1]) *
                    CFG.refinement_data_low_level_execution_cost)
            # Save the trajectory, e.g., for playback on a robot.
            traj_file = f"{save_prefix}__task{test_task_idx+1}.traj"
            traj_file_path = Path(CFG.eval_trajectories_dir) / traj_file
            # Include the original task too so we know the goal.
            traj_data = {"task": env_task, "trajectory": traj}
            with open(traj_file_path, "wb") as f:
                pkl.dump(traj_data, f)
        except utils.EnvironmentFailure as e:
            log_message = f"Environment failed with error: {e}"
            caught_exception = True
        except (ApproachTimeout, ApproachFailure) as e:
            log_message = ("Approach failed at policy execution time with "
                           f"error: {e}")
            if isinstance(e, ApproachTimeout):
                total_num_execution_timeouts += 1
            elif isinstance(e, ApproachFailure):
                total_num_execution_failures += 1
            caught_exception = True
        if solved:
            log_message = "SOLVED"
            num_solved += 1
            total_suc_time += (solve_time + exec_time)
            make_video = CFG.make_test_videos
            video_file = f"{save_prefix}__task{test_task_idx+1}.mp4"
        else:
            if not caught_exception:
                log_message = "Policy failed to reach goal"
            if CFG.crash_on_failure:
                raise RuntimeError(log_message)
            make_video = CFG.make_failure_videos
            video_file = f"{save_prefix}__task{test_task_idx+1}_failure.mp4"
        logging.info(f"Task {test_task_idx+1} / {len(test_tasks)}: "
                     f"{log_message}")
        if make_video:
            assert monitor is not None
            video = monitor.get_video()
            utils.save_video(video_file, video)
    metrics["num_solved"] = num_solved
    metrics["num_total"] = len(test_tasks)
    metrics["avg_suc_time"] = (total_suc_time /
                               num_solved if num_solved > 0 else float("inf"))
    metrics["avg_ref_cost"] = ((total_low_level_action_cost +
                                cogman.metrics["total_refinement_time"]) /
                               num_solved if num_solved > 0 else float("inf"))
    metrics["min_num_samples"] = cogman.metrics[
        "min_num_samples"] if cogman.metrics["min_num_samples"] < float(
            "inf") else 0
    metrics["max_num_samples"] = cogman.metrics["max_num_samples"]
    metrics["min_skeletons_optimized"] = cogman.metrics[
        "min_num_skeletons_optimized"] if cogman.metrics[
            "min_num_skeletons_optimized"] < float("inf") else 0
    metrics["max_skeletons_optimized"] = cogman.metrics[
        "max_num_skeletons_optimized"]
    metrics["num_solve_timeouts"] = total_num_solve_timeouts
    metrics["num_solve_failures"] = total_num_solve_failures
    metrics["num_execution_timeouts"] = total_num_execution_timeouts
    metrics["num_execution_failures"] = total_num_execution_failures
    # Handle computing averages of total cogman metrics wrt the
    # number of found policies. Note: this is different from computing
    # an average wrt the number of solved tasks, which might be more
    # appropriate for some metrics, e.g. avg_suc_time above.
    for metric_name in [
            "num_samples", "num_skeletons_optimized", "num_nodes_expanded",
            "num_nodes_created", "num_nsrts", "num_preds", "plan_length",
            "num_failures_discovered"
    ]:
        total = cogman.metrics[f"total_{metric_name}"]
        metrics[f"avg_{metric_name}"] = (
            total / num_found_policy if num_found_policy > 0 else float("inf"))
    return metrics


def _save_test_results(results: Metrics) -> None:
    num_solved = results["num_solved"]
    num_total = results["num_total"]
    avg_suc_time = results["avg_suc_time"]
    logging.info(f"Tasks solved: {num_solved} / {num_total}")
    logging.info(f"Average time for successes: {avg_suc_time:.5f} seconds")
    outfile = f"{CFG.results_dir}/{utils.get_config_path_str()}.pkl"
    # Save CFG alongside results.
    outdata = {
        "config": CFG,
        "results": results.copy(),
        "git_commit_hash": utils.get_git_commit_hash()
    }
    # Dump the CFG, results, and git commit hash to a pickle file.
    with open(outfile, "wb") as f:
        pkl.dump(outdata, f)
    # Before printing the results, filter out keys that start with the
    # special prefix "PER_TASK_", to prevent an annoyingly long printout.
    del_keys = [k for k in results if k.startswith("PER_TASK_")]
    for k in del_keys:
        del results[k]
    logging.info(f"Test results: {results}")
    logging.info(f"Wrote out test results to {outfile}")


if __name__ == "__main__":  # pragma: no cover
    # Write out the exception to the log file.
    mp.set_start_method('spawn', force=True)
    try:
        main()
    except Exception as _err:  # pylint: disable=broad-except
        logging.exception("main.py crashed")
        raise _err
