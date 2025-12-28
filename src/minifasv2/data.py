import os
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import pandas as pd
import math


def load_image(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def generate_FT(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift) + 1)
    maxx = -1
    minn = 100000
    for i in range(len(fimg)):
        if maxx < max(fimg[i]):
            maxx = max(fimg[i])
        if minn > min(fimg[i]):
            minn = min(fimg[i])
    fimg = (fimg - minn + 1) / (maxx - minn + 1)
    return fimg


class LivenessDataset(Dataset):
    def __init__(
        self, root, labels, transform=None, target_transform=None, reader=load_image
    ):
        self.root = root
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.reader = reader

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.labels.iloc[idx, 0])
        sample = self.reader(path)
        target = self.labels.iloc[idx, 1]

        if sample is None:
            print("image is None --> ", path)
        assert sample is not None

        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except Exception as err:
                print("Error Occured: %s" % err, path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


class LivenessDatasetFT(LivenessDataset):
    def __init__(
        self,
        root,
        labels,
        transform=None,
        target_transform=None,
        reader=load_image,
        fourier_size=(10, 10),
    ):
        super().__init__(root, labels, transform, target_transform, reader)
        self.fourier_size = fourier_size

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.labels.iloc[idx, 0])
        sample = self.reader(path)
        target = self.labels.iloc[idx, 1]

        fourier_sample = generate_FT(sample)
        if sample is None:
            print("image is None --> ", path)
        if fourier_sample is None:
            print("FT image is None --> ", path)
        assert sample is not None

        fourier_sample = cv2.resize(fourier_sample, self.fourier_size)
        fourier_sample = torch.from_numpy(fourier_sample).float()
        fourier_sample = torch.unsqueeze(fourier_sample, 0)

        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except Exception as err:
                print("Error occured: %s" % err, path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, fourier_sample, target


class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [
            max_wh - (s + pad) for s, pad in zip(image.size, [p_left, p_top])
        ]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, "constant")


class RandomRotationWithReflect:
    def __init__(self, degrees, expand=False):
        self.degrees = degrees
        self.expand = expand

    def __call__(self, img):
        angle = T.RandomRotation.get_params([-self.degrees, self.degrees])

        if isinstance(img, Image.Image):
            img_np = np.array(img, dtype=np.uint8)
            if len(img_np.shape) == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        else:
            img_np = np.array(img, dtype=np.uint8)

        h, w = img_np.shape[:2]
        center = (w // 2, h // 2)

        if self.expand:
            diagonal = math.sqrt(w**2 + h**2)
            new_w = int(diagonal)
            new_h = int(diagonal)

            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))

            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]

            img_rotated = cv2.warpAffine(
                img_np,
                M,
                (new_w, new_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            return Image.fromarray(img_rotated, "RGB")
        else:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            img_rotated = cv2.warpAffine(
                img_np,
                M,
                (w, h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            return Image.fromarray(img_rotated, "RGB")


def transform_labels(labels, categories):
    def spoof_transform(t):
        return next(i for i, category in enumerate(categories) if t in category)

    return labels.apply(spoof_transform)


def transform_spoof_type(spoof_type, spoof_categories):
    for idx, category in enumerate(spoof_categories):
        if spoof_type in category:
            return idx
    return None


def load_labels_from_json(json_path, processed_dir, spoof_categories, split="train"):
    train_label = pd.read_json(json_path, orient="index").apply(
        pd.to_numeric, downcast="integer"
    )

    spoof_filter = [item for sublist in spoof_categories for item in sublist]
    train_label = train_label[train_label[40].isin(spoof_filter)]

    train_label.index = train_label.index.str.replace("Data/", "")

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = set()
    if processed_dir:
        for root, dirs, files in os.walk(os.path.join(processed_dir, split)):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    rel_path = os.path.relpath(
                        os.path.join(root, file), os.path.join(processed_dir, split)
                    )
                    image_files.add(rel_path.replace("\\", "/"))

    data = []
    for idx, spoof_type in train_label[40].items():
        path = idx.replace(f"{split}/", "") if idx.startswith(f"{split}/") else idx
        if image_files and path not in image_files:
            continue
        mapped_type = transform_spoof_type(spoof_type, spoof_categories)
        if mapped_type is not None:
            data.append({"path": path, "spoof_type": mapped_type})

    return pd.DataFrame(data)


def get_train_valid(config):

    train_transform = T.Compose(
        [
            T.ToPILImage(),
            T.RandomResizedCrop(size=tuple(2 * [config.input_size]), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            RandomRotationWithReflect(90),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]
    )

    valid_transform = T.Compose([T.ToPILImage(), T.ToTensor()])

    processed_dir = os.path.dirname(os.path.dirname(config.labels_path))
    train_labels = load_labels_from_json(
        config.labels_path, processed_dir, config.spoof_categories, "train"
    )

    if config.spoof_categories is not None:
        unique_labels = set(train_labels.iloc[:, 1].unique())
        original_labels = set(
            [item for sublist in config.spoof_categories for item in sublist]
        )
        transformed_labels = set(range(len(config.spoof_categories)))
        original_only = original_labels - transformed_labels

        if unique_labels.intersection(original_only):
            train_labels.iloc[:, 1] = transform_labels(
                train_labels.iloc[:, 1], config.spoof_categories
            )
    if config.class_balancing is not None:
        cb = config.class_balancing
        if cb == "down":
            value_counts = train_labels.iloc[:, 1].value_counts()
            train_downsampled = [
                train_labels[train_labels.iloc[:, 1] == value_counts.index[-1]]
            ]
            for value in value_counts.index[:-1]:
                train_downsampled.append(
                    train_labels[train_labels.iloc[:, 1] == value].sample(
                        value_counts.min()
                    )
                )
            train_labels = pd.concat(train_downsampled)

    train_labels, valid_labels = train_test_split(
        train_labels, test_size=config.valid_size, random_state=20220826
    )

    train_labels = train_labels.reset_index(drop=True)
    valid_labels = valid_labels.reset_index(drop=True)

    train_data = DataLoader(
        LivenessDatasetFT(
            config.train_path,
            train_labels,
            train_transform,
            None,
            fourier_size=config.fourier_size,
        ),
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    valid_data = DataLoader(
        LivenessDataset(config.train_path, valid_labels, valid_transform, None),
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    return train_data, valid_data


def get_test(config):

    test_transform = T.Compose([T.ToPILImage(), SquarePad(), T.ToTensor()])

    processed_dir = os.path.dirname(os.path.dirname(config.labels_path))
    test_labels = load_labels_from_json(
        config.labels_path, processed_dir, config.spoof_categories, "test"
    )

    if config.spoof_categories is not None:
        unique_labels = set(test_labels.iloc[:, 1].unique())
        original_labels = set(
            [item for sublist in config.spoof_categories for item in sublist]
        )
        transformed_labels = set(range(len(config.spoof_categories)))
        original_only = original_labels - transformed_labels

        if unique_labels.intersection(original_only):
            test_labels.iloc[:, 1] = transform_labels(
                test_labels.iloc[:, 1], config.spoof_categories
            )
    test_data = DataLoader(
        LivenessDataset(config.test_path, test_labels, test_transform, None),
        batch_size=config.batch_size,
        pin_memory=True,
    )

    return test_data
