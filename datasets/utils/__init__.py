

def get_mvtec_classes(args):
    if args.dataset.dataset_order == 1:
        mvtec_classes = ['leather', 'bottle', 'metal_nut',
                         'grid', 'screw', 'zipper',
                         'tile', 'hazelnut', 'toothbrush',
                         'wood', 'transistor', 'pill',
                         'carpet', 'capsule', 'cable']
    elif args.dataset.dataset_order == 2:
        mvtec_classes = ['wood', 'transistor', 'pill',
                         'tile', 'hazelnut', 'toothbrush',
                         'leather', 'bottle', 'metal_nut',
                         'carpet', 'capsule', 'cable',
                         'grid', 'screw', 'zipper']
    elif args.dataset.dataset_order == 3:
        mvtec_classes = ['leather', 'grid', 'tile',
                         'bottle', 'toothbrush', 'capsule',
                         'screw', 'pill', 'zipper',
                         'cable', 'metal_nut', 'hazelnut',
                         'wood', 'carpet', 'transistor']
    return mvtec_classes


def get_visa_classes(args):
    if args.dataset.dataset_order == 1:
        visa_classes = ['candle', 'capsules', 'cashew',
                        'chewinggum', 'fryum', 'macaroni1',
                        'macaroni2', 'pcb1', 'pcb2',
                        'pcb3', 'pcb4', 'pipe_fryum']
    elif args.dataset.dataset_order == 2:
        visa_classes = ['pcb1', 'pcb2', 'pcb3',
                        'pcb4', 'pipe_fryum', 'candle',
                        'capsules', 'cashew', 'chewinggum',
                        'fryum', 'macaroni1', 'macaroni2']
    elif args.dataset.dataset_order == 3:
        visa_classes = ['cashew', 'fryum', 'pcb3',
                        'candle', 'macaroni1', 'pcb1',
                        'capsules', 'chewinggum', 'pcb4',
                        'macaroni2', 'pipe_fryum', 'pcb2']
    return visa_classes
