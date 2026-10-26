"""UniPred settings, registered into the predicators global settings.

Like predicators.settings.GlobalSettings, every attribute is a default
that can be overridden on the command line (``--llm_model gpt-4o``) or in
an experiment config (see unipred/config.py).
"""

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
    # Number of demonstrations shown in the PDDL-completion prompt.
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
    # Table Clean: number of toys in train and out-of-distribution test
    # tasks.
    table_clean_num_toys_train = 2
    table_clean_num_toys_test = 3

    # Validation loss that maps to score 0 in the LLM feedback; the
    # acceptance threshold (loss_thresh of the predicate config) maps
    # to 100, on a log scale in between.
    unipred_score_zero_loss = 1.0


register_settings(UniPredSettings)
