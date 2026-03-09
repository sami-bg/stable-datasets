import io
import tarfile
from pathlib import Path

from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Image, Version
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, download


class Imagenet(BaseDatasetBuilder):
    """ImageNet train archive loader compatible with the existing builder paradigm."""

    VERSION = Version("2.0.0")
    SOURCE = {
        "homepage": "https://www.image-net.org/challenges/LSVRC/2012/",
        "assets": {
            "train": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar",
        },
        "citation": """@article{deng2009imagenet,
        title={ImageNet: A large-scale hierarchical image database},
        author={Deng, Jia and others},
        journal={CVPR},
        year={2009}
    }""",
    }

    def __init__(self, streaming: bool = True, class_limit: int | None = None, **kwargs):
        self.streaming = streaming
        self.class_limit = class_limit
        super().__init__(**kwargs)

    def _info(self):
        class_count = self.class_limit if self.class_limit is not None else 1000
        width = max(2, len(str(class_count - 1)))
        class_names = [f"class_{idx:0{width}d}" for idx in range(class_count)]

        return DatasetInfo(
            description="ImageNet train split in TAR format with optional streaming mode.",
            features=Features({"image": Image(), "label": ClassLabel(names=class_names)}),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self, dl_manager=None):
        train_path = download(self.SOURCE["assets"]["train"], dest_folder=self._raw_download_dir)
        return [SplitGenerator(name=Split.TRAIN, gen_kwargs={"data_path": train_path})]

    def _generate_examples(self, data_path, split=None):
        mode = "r|*" if self.streaming else "r:*"
        class_count = 0

        with tarfile.open(Path(data_path), mode) as outer:
            for member in outer:
                if not member.isfile() or not member.name.endswith(".tar"):
                    continue
                if self.class_limit is not None and class_count >= self.class_limit:
                    break

                class_file = outer.extractfile(member)
                if class_file is None:
                    continue

                with tarfile.open(fileobj=io.BytesIO(class_file.read()), mode="r:*") as inner:
                    for image_member in inner:
                        if not image_member.isfile():
                            continue
                        if not image_member.name.lower().endswith((".jpg", ".jpeg", ".png")):
                            continue

                        image_file = inner.extractfile(image_member)
                        if image_file is None:
                            continue

                        image = PILImage.open(io.BytesIO(image_file.read())).convert("RGB")
                        key = f"{Path(member.name).stem}/{image_member.name}"
                        yield key, {"image": image, "label": class_count}

                class_count += 1
