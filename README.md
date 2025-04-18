## YOLO-World YOLOv9
### without weight (use the first version)
| model | Schedule  | `mask-refine` | efficient neck |  AP | AP<sub>50</sub> | AP<sub>75</sub> | weights | log35 |
| :---- | :-------: | :----------: |:-------------:  | :-: | :--------------:| :-------------: |:------: | :-: |
| [YOLO-World-v2-L (v1)](./yolo_world_v2_l_vlpan_bn_sgd_1e-3_40e_8gpus_finetune_coco.py)  | SGD, 1e-3, 40e | ✖️  | ✖️ | 37.4 | 52.7 | 40.8 | [HF Checkpoints](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_vlpan_bn_sgd_1e-3_40e_8gpus_finetune_coco_ep80-e1288152.pth) | [log](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_vlpan_bn_sgd_1e-3_40e_8gpus_finetuning_coco_20240327_014902.log) |

### with weight
| model | Schedule  | `mask-refine` | efficient neck |  AP | AP<sub>50</sub> | AP<sub>75</sub> | weights | logOri3 |
| :---- | :-------: | :----------: |:-------------:  | :-: | :--------------:| :-------------: |:------: | :-: |
| [YOLO-World-v2-L (v1)](./yolo_world_v2_l_vlpan_bn_sgd_1e-3_40e_8gpus_finetune_coco.py)  | SGD, 1e-3, 40e | ✖️  | ✖️ | 37.8 | 53 | 41 | [HF Checkpoints](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_vlpan_bn_sgd_1e-3_40e_8gpus_finetune_coco_ep80-e1288152.pth) | [log](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_vlpan_bn_sgd_1e-3_40e_8gpus_finetuning_coco_20240327_014902.log) |

### with weight
| model | Schedule  | `mask-refine` | efficient neck |  AP | AP<sub>50</sub> | AP<sub>75</sub> | weights | logOri4 |
| :---- | :-------: | :----------: |:-------------:  | :-: | :--------------:| :-------------: |:------: | :-: |
| [YOLO-World-v2-L (v1)](./yolo_world_v2_l_vlpan_bn_sgd_1e-3_40e_8gpus_finetune_coco.py)  | SGD, 1e-3, 40e | ✖️  | ✖️ | 38.1 | 53.5 | 41.5 | [HF Checkpoints](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_vlpan_bn_sgd_1e-3_40e_8gpus_finetune_coco_ep80-e1288152.pth) | [log](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_vlpan_bn_sgd_1e-3_40e_8gpus_finetuning_coco_20240327_014902.log) |

### YOLOv9 With CBAM
| model | Schedule  | `mask-refine` | efficient neck |  AP | AP<sub>50</sub> | AP<sub>75</sub> | weights | logOri5 |
| :---- | :-------: | :----------: |:-------------:  | :-: | :--------------:| :-------------: |:------: | :-: |
| [YOLO-World-v2-L (v1)](./yolo_world_v2_l_vlpan_bn_sgd_1e-3_40e_8gpus_finetune_coco.py)  | SGD, 1e-3, 40e | ✖️  | ✖️ | 38.6 | 54.0 | 42.2 | [HF Checkpoints](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_vlpan_bn_sgd_1e-3_40e_8gpus_finetune_coco_ep80-e1288152.pth) | [log](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_vlpan_bn_sgd_1e-3_40e_8gpus_finetuning_coco_20240327_014902.log) |

