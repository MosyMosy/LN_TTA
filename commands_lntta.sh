#!/bin/bash

# Base arguments
DEVICE=0
METHOD=tta_cls_stat
BATCH_SIZE=32
BATCH_SIZE_TTA=1 # This is for augmentation batch size

# Dataset configs and checkpoints
declare -A datasets=(
  [scanobject]="cfgs/tta_purge/tta_purge_scanobject.yaml checkpoints/scan_object_src_only.pth simple"
  [modelnet]="cfgs/tta_purge/tta_purge_modelnet.yaml checkpoints/modelnet_src_only.pth full"
  [shapenet]="cfgs/tta_purge/tta_purge_shapenet.yaml checkpoints/shapenet_src_only.pth full"
# )

# Modes and exp_name suffixes (always 3 fields: cls_fixer_mode, BN_reset_flag, exp_name)
modes=(
  "source_only '' source_only_noBNreset"
  "source_only --BN_reset source_only_BNreset"
  "source_only_cls-fixer '' source_only_ClsFixer_noBNreset"
  "source_only_cls-fixer --BN_reset source_only_ClsFixer_BNreset"
  "update_tent '' Tent_noBNreset"
  "update_tent --BN_reset Tent_BNreset"
  "update_tent_cls-fixer '' Tent_ClsFixer_noBNreset"
  "update_tent_cls-fixer --BN_reset Tent_ClsFixer_BNreset"
)

# Online modes ("" means no --online, "--online" means online mode)
online_flags=("--online")

# Loop over online flag, modes, and datasets
for online_flag in "${online_flags[@]}"; do
  for mode_info in "${modes[@]}"; do
    read -r cls_mode bn_flag exp_name <<< "$mode_info"
    exp_name="${exp_name}_bs${BATCH_SIZE}"
    [[ "$bn_flag" == "''" ]] && bn_flag=""

    # Add prefix if online
    if [[ "$online_flag" == "--online" ]]; then
      exp_name="online_${exp_name}"
    fi

    for name in "${!datasets[@]}"; do
      read -r config ckpt ln_mode <<< "${datasets[$name]}"
      echo "Running $exp_name on $name with online=$online_flag"

      CUDA_VISIBLE_DEVICES=$DEVICE python main.py \
        --config "$config" \
        --ckpts "$ckpt" \
        --method "$METHOD" \
        --batch_size $BATCH_SIZE \
        --batch_size_tta $BATCH_SIZE_TTA \
        --cls_fixer_mode "$cls_mode" \
        --cls_fixer_ln_mode "$ln_mode" \
        $bn_flag \
        $online_flag \
        --exp_name "$exp_name"
    done
  done
done
