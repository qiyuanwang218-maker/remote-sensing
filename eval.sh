model_path="./experiments/rag_7M_gpt2"
val_annotations="./data/val_gt.json"
test_annotations="./data/test_gt.json"


for checkpoint_dir in ${model_path}/checkpoint-*; do
    if [ -d "$checkpoint_dir" ]; then

        if [ -f "${checkpoint_dir}/val_preds.json" ]; then
            echo "Running eval for validation with checkpoint: $checkpoint_dir"
            python coco-caption/run_eval.py $val_annotations "${checkpoint_dir}/val_preds.json"
        else
            echo "val_preds.json not found for checkpoint: $checkpoint_dir"
        fi

        if [ -f "${checkpoint_dir}/test_preds.json" ]; then
            echo "Running eval for test with checkpoint: $checkpoint_dir"
            python coco-caption/run_eval.py $test_annotations "${checkpoint_dir}/test_preds.json"
        else
            echo "test_preds.json not found for checkpoint: $checkpoint_dir"
        fi
    fi
done
