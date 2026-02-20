# UniPred

## Running UniPred
### Prerequisites

 Create a `.env.local` file in the project root containing your OpenAI API key:
 OPENAI_API_KEY=your_openai_api_key_here

### Satellites Environment

To run UniPred on the satellites environment:

1. **Training**: Train the model from scratch using LLM-guided search
   ```bash
   mkdir -p logs/satellites
   bash ./scripts/train/satellites/unipred.sh
   ```

2. **Testing**: Evaluate the trained model
   ```bash
   mkdir -p logs/satellites  
   bash ./scripts/test/satellites/unipred.sh
   ```



### Other Environments

To adapt IVNTR-LLMSearch to other environments, you need to create two configuration files similar to the satellites setup:

1. **Create PDDL configuration** (`predicators/config/<env>/pddl.json`):
   ```json
   {
     "version": 1,
     "predicates": [
       {
         "name": "<predicate_name>",
         "arity": <number_of_arguments>,
         "types": ["<type1>", "<type2>"],
         "role": "goal|precondition",
         "effect_map": {
           "<action_name>": "add|del"
         }
       }
     ]
   }
   ```

2. **Create neural predicate configuration** (`predicators/config/<env>/pred_pdlm.yaml`):
   ```yaml
   final_op: [0, 0, 0, 0, 0, 1, 1, 1]  # Binary vector indicating operations
   neupi_non_effect_predicates: ['<predicate1>', '<predicate2>']  # Fixed predicates

   config:
     - name: "<predicate_name>"
       types: ["<type1>", "<type2>"]
       num_vectors_to_generate: <number>
       ent_idx: [0, 1]  # Entity indices
       architecture:
         type: "MLP"
         layer_size: 32|128
       optimizer:
         type: "AdamW"
         kwargs:
           lr: 0.001
       batch_size: 512
       epochs: 100|200
       num_iter: 5|70
       guidance_thresh: 0.05|0.1
       loss_thresh: 0.005
   ```

3. **Create training script** following the pattern of `unipred.sh`:
   ```bash
   python3 predicators/main.py --env <your_env> --approach unipred \
       --neupi_pred_config "predicators/config/<env>/pred_pdlm.yaml" \
       --pred_pddl_config "predicators/config/<env>/pddl.json" \