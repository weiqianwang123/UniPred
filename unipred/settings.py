"""UniPred settings, registered into the predicators global settings.

Like predicators.settings.GlobalSettings, every attribute is a default
that can be overridden on the command line (``--llm_model gpt-4o``) or in
an experiment config (see unipred/config.py).
"""

import os

from predicators.settings import register_settings


class UniPredSettings:
    """Settings of the UniPred approaches and LLM client."""
    # LLM client (unipred/llm/client.py).
    # One of "openai", "qwen", "gemini", "codex" (the local Codex CLI) or
    # "mock" (offline, for tests).
    llm_provider = "openai"
    llm_model = "gpt-4o"
    # Not used by the codex provider, which has no temperature setting.
    llm_temperature = 0.2
    llm_max_tokens = 4096
    llm_timeout = 120.0
    llm_retry_attempts = 3
    # Overrides the provider's default endpoint (e.g. for Qwen regions).
    llm_base_url = ""
    # Reasoning effort of the codex provider ("low", "medium", ...).
    llm_reasoning_effort = "low"
    # Command that runs the Codex CLI.
    llm_codex_command = "codex"
    # Responses are cached on disk, keyed by the request, so that reruns
    # are reproducible and free. An empty string disables the cache.
    llm_cache_dir = "outputs/llm_cache"
    # JSON file of canned responses for the mock provider (see
    # unipred/llm/client.py); empty means rule-based mock responses.
    llm_mock_responses = ""

    # The domain's description and known predicates come from the predicate
    # config (--neupi_pred_config; see configs/satellites/predicates.yaml).
    # Number of demonstrations shown in the LLM prompts.
    unipred_num_demo_prompts = 3

    # UniPred stages; turning one off gives the ablations of the paper.
    # Stage 2: seed predicates from LLM PDDL completion.
    unipred_pddl_completion = True
    # Stage 3: LLM-in-the-loop effect proposals with classifier feedback.
    unipred_llm_loop = True
    # Show the validation scores of earlier proposals to the LLM.
    unipred_loss_feedback = True
    # Show operator and known predicate names (False: neutral ids).
    unipred_semantic_names = True
    # Stage 4: derived-aware predicate selection (Algorithm 1).
    unipred_derived_selection = False

    # Code as Policy (unipred/approaches/code_as_policy.py): how often the
    # LLM may repair a policy that fails with an error on the training
    # tasks, and the time limit of one call of the policy, in seconds.
    code_as_policy_max_repairs = 3
    code_as_policy_call_timeout = 1.0

    # ViLa (unipred/approaches/vila.py): "zero_shot", "few_shot" (with
    # demonstrations) or "hpe" (few_shot with the hand-written constraints
    # in the text file vila_constraints).
    vila_mode = "zero_shot"
    vila_constraints = ""

    # Pix2Pred (unipred/approaches/pix2pred.py): use the names and types of
    # the environment's hidden ground-truth predicates instead of invented
    # ones, and take the atoms of these predicates (comma-separated names)
    # from the annotations instead of the VLM.
    pix2pred_gt_predicates = False
    pix2pred_gt_label_predicates = ""

    # Threads of torch in the main process. The models are small, and
    # torch's default of one thread per core makes them train many times
    # slower when other processes share the CPU (the fork's parallel
    # classifier training divides the cores among its processes).
    torch_num_threads = 1

    # Perception (unipred/perception): the directory that scripts/install.sh
    # downloads the weights into (default: $UNIPRED_WEIGHTS_DIR), and the
    # encoder of object crops ("dinov3", "resnet18", "fake" for tests, or
    # "auto": DINOv3 if installed, else ResNet-18).
    perception_weights_dir = os.environ.get("UNIPRED_WEIGHTS_DIR", "")
    perception_encoder = "auto"

    # Packing: numbers of boxes and of large and small items in train and
    # out-of-distribution test tasks.
    packing_num_boxes_train = 2
    packing_num_large_train = 1
    packing_num_small_train = 2
    packing_num_boxes_test = 3
    packing_num_large_test = 2
    packing_num_small_test = 2

    # Table Clean Real: the output directory of
    # examples/real_robot/process_demos.py.
    table_clean_real_data_dir = ""

    # Table Clean Sim: number of toys in train and out-of-distribution test
    # tasks.
    table_clean_sim_num_toys_train = 2
    table_clean_sim_num_toys_test = 3

    # Validation loss that maps to score 0 in the LLM feedback; the
    # acceptance threshold (loss_thresh of the predicate config) maps
    # to 100, on a log scale in between.
    unipred_score_zero_loss = 1.0


register_settings(UniPredSettings)
