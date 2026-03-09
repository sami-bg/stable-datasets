import io
import tarfile

from PIL import Image

from stable_datasets.images.imagenet import Imagenet
from stable_datasets.images.imagenette import Imagenette


def _jpeg_bytes(color=(255, 0, 0)):
    img = Image.new("RGB", (8, 8), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _create_imagenet_train_tar(path, num_classes=3, images_per_class=2):
    with tarfile.open(path, "w") as outer:
        for idx in range(num_classes):
            class_name = f"n{idx:08d}"
            class_buf = io.BytesIO()
            with tarfile.open(fileobj=class_buf, mode="w") as inner:
                for j in range(images_per_class):
                    payload = _jpeg_bytes(color=(idx, j, 0))
                    info = tarfile.TarInfo(name=f"{class_name}_{j}.JPEG")
                    info.size = len(payload)
                    inner.addfile(info, io.BytesIO(payload))
            class_tar = class_buf.getvalue()
            outer_info = tarfile.TarInfo(name=f"{class_name}.tar")
            outer_info.size = len(class_tar)
            outer.addfile(outer_info, io.BytesIO(class_tar))


def _create_imagenette_tar(path):
    with tarfile.open(path, "w:gz") as archive:
        classes = ["n01440764", "n02102040"]
        for split in ["train", "val"]:
            for cls in classes:
                payload = _jpeg_bytes()
                name = f"imagenette2/{split}/{cls}/{cls}_{split}.JPEG"
                info = tarfile.TarInfo(name=name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))


def test_imagenet_streaming_integration(tmp_path, monkeypatch):
    tar_path = tmp_path / "ILSVRC2012_img_train.tar"
    _create_imagenet_train_tar(tar_path, num_classes=2, images_per_class=2)
    monkeypatch.setattr("stable_datasets.images.imagenet.download", lambda *args, **kwargs: tar_path)

    ds = Imagenet(split="train", streaming=True, processed_cache_dir=tmp_path / "processed")
    assert len(ds) == 4
    sample = ds[0]
    assert isinstance(sample["image"], Image.Image)
    assert 0 <= sample["label"] < 1000


def test_imagenette_integration(tmp_path, monkeypatch):
    tar_path = tmp_path / "imagenette2.tgz"
    _create_imagenette_tar(tar_path)
    monkeypatch.setattr("stable_datasets.images.imagenette.download", lambda *args, **kwargs: tar_path)

    train = Imagenette(split="train", processed_cache_dir=tmp_path / "processed")
    test = Imagenette(split="test", processed_cache_dir=tmp_path / "processed")

    assert len(train) == 2
    assert len(test) == 2
    assert isinstance(train[0]["image"], Image.Image)
    assert 0 <= train[0]["label"] < 10
