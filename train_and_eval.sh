#!/bin/bash

# Boucles pour chaque combinaison de stratégie et taux d'augmentation
for aug_rate in 0.1 1.0; do
    for aug_strategy in conll wgold morrowind dekker; do
        echo "Training with augmentation strategy: ${aug_strategy}, rate: ${aug_rate}"

        # Entraînement
        python train.py \
            --epochs-nb 2 \
            --batch-size 6 \
            --context-size 1 \
            --data-aug-strategies "{\"PER\": [\"${aug_strategy}\"]}" \
            --data-aug-frequencies "{\"PER\": [${aug_rate}]}" \
            --model-path "augmented_model_${aug_strategy}_${aug_rate}.pth"

        echo "Evaluating model: augmented_model_${aug_strategy}_${aug_rate}.pth"

        # Évaluation
        python extract_metrics.py \
            --model-path "augmented_model_${aug_strategy}_${aug_rate}.pth" \
            --global-metrics \
            --context-size 1 \
            --book-group "fantasy" \
            --output-file "results_${aug_strategy}_${aug_rate}.json"
    done
done
