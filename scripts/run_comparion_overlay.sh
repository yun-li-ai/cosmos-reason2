    python3 scripts/critique_layered_control_alignment.py \
        --layered-dir assets/overlay_output/b_trained_golden_hour \
        --auto-multiview-json assets/inference_results/b_trained_golden_hour/auto_multiview.json \
        --output assets/overlay_output/critique_results2/critique_b_trained_golden_hour.md

    python3 scripts/critique_layered_control_alignment.py \
        --layered-dir assets/overlay_output/a_trained_golden_hour \
        --auto-multiview-json assets/inference_results/a_trained_golden_hour/auto_multiview.json \
        --output assets/overlay_output/critique_results2/critique_a_trained_golden_hour.md

    python3 scripts/critique_layered_control_alignment.py \
        --layered-dir assets/overlay_output/b_trained_night \
        --auto-multiview-json assets/inference_results/b_trained_night/auto_multiview.json \
        --output assets/overlay_output/critique_results2/critique_b_trained_night.md

    python3 scripts/critique_layered_control_alignment.py \
        --layered-dir assets/overlay_output/a_trained_night \
        --auto-multiview-json assets/inference_results/a_trained_night/auto_multiview.json \
        --output assets/overlay_output/critique_results2/critique_a_trained_night.md

    python3 scripts/critique_layered_control_alignment.py \
        --layered-dir assets/overlay_output/a_trained_snowy \
        --auto-multiview-json assets/inference_results/a_trained_snowy/auto_multiview.json \
        --output assets/overlay_output/critique_results2/critique_a_trained_snowy.md

    python3 scripts/critique_layered_control_alignment.py \
        --layered-dir assets/overlay_output/b_trained_snowy \
        --auto-multiview-json assets/inference_results/b_trained_snowy/auto_multiview.json \
        --output assets/overlay_output/critique_results2/critique_b_trained_snowy.md
