python scripts/critique_compare_model_ab.py   --control-dir assets/inference_results/control_videos   --model-a-dir assets/inference_results/a_trained_night   --model-b-dir assets/inference_results/b_trained_night   --auto-multiview-json assets/inference_results/b_trained_night/auto_multiview.json --output assets/inference_results/critique_results2/critique_night.md

python scripts/critique_compare_model_ab.py   --control-dir assets/inference_results/control_videos   --model-a-dir assets/inference_results/a_trained_snowy   --model-b-dir assets/inference_results/b_trained_snowy   --auto-multiview-json assets/inference_results/b_trained_snowy/auto_multiview.json --output assets/inference_results/critique_results2/critique_snowy.md

python scripts/critique_compare_model_ab.py   --control-dir assets/inference_results/control_videos   --model-a-dir assets/inference_results/a_trained_golden_hour   --model-b-dir assets/inference_results/b_trained_golden_hour   --auto-multiview-json assets/inference_results/b_trained_golden_hour/auto_multiview.json --output assets/inference_results/critique_results2/critique_golden_hour.md



python scripts/critique_compare_model_ab.py   --control-dir assets/inference_results/control_videos   --model-b-dir assets/inference_results/a_trained_night   --model-a-dir assets/inference_results/b_trained_night   --auto-multiview-json assets/inference_results/b_trained_night/auto_multiview.json --output assets/inference_results/critique_results_ba2/critique_night.md

python scripts/critique_compare_model_ab.py   --control-dir assets/inference_results/control_videos   --model-b-dir assets/inference_results/a_trained_snowy   --model-a-dir assets/inference_results/b_trained_snowy   --auto-multiview-json assets/inference_results/b_trained_snowy/auto_multiview.json --output assets/inference_results/critique_results_ba2/critique_snowy.md

python scripts/critique_compare_model_ab.py   --control-dir assets/inference_results/control_videos   --model-b-dir assets/inference_results/a_trained_golden_hour   --model-a-dir assets/inference_results/b_trained_golden_hour   --auto-multiview-json assets/inference_results/b_trained_golden_hour/auto_multiview.json --output assets/inference_results/critique_results_ba2/critique_golden_hour.md