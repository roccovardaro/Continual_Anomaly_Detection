from torchvision import transforms
from .transforms import aug_transformation, no_aug_transformation
from .visa_dataset import VisADataset
from torch.utils.data import DataLoader
from .utils import get_visa_classes


def get_visa_dataloaders(args, t, dataloaders_train, dataloaders_test, learned_tasks, all_test_filenames):
    visa_classes = get_visa_classes(args)

    N_CLASSES_PER_TASK = args.dataset.n_classes_per_task
    if args.dataset.data_incre_setting == 'one':
        # Scenario incrementale: primo task con bulk di classi, poi una alla volta
        if t == 0:
            task_visa_classes = visa_classes[: 8]  # VisA ha 12 classi, primo task con 8
        else:
            i = 8 + (t - 1) * N_CLASSES_PER_TASK
            task_visa_classes = visa_classes[i: i + N_CLASSES_PER_TASK]
    else:
        # Scenario multi-class incrementale: n classi per task
        i = t * N_CLASSES_PER_TASK
        task_visa_classes = visa_classes[i: i + N_CLASSES_PER_TASK]
    learned_tasks.append(task_visa_classes)

    train_transform = aug_transformation(args)
    test_transform = no_aug_transformation(args)

    train_data = VisADataset(args.data_dir, task_visa_classes, transform=train_transform, size=args.dataset.image_size)
    test_data = VisADataset(args.data_dir, task_visa_classes, args.dataset.image_size, transform=test_transform, mode="test")
    all_test_filenames.append(test_data.all_image_names)

    train_dataloader = DataLoader(train_data, batch_size=args.train.batch_size, shuffle=True, num_workers=args.dataset.num_workers)
    dataloaders_train.append(train_dataloader)
    dataloader_test = DataLoader(test_data, batch_size=args.eval.batch_size, shuffle=False, num_workers=args.dataset.num_workers)
    dataloaders_test.append(dataloader_test)
    print('class name:', task_visa_classes, 'number of training sets:', len(train_data),
          'number of testing sets:', len(test_data))

    return train_dataloader, dataloaders_train, dataloaders_test, learned_tasks, len(train_data), all_test_filenames
