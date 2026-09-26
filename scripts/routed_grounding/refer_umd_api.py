# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small Python 3 implementation of the official REFER read API contract.

The upstream ``lichengunc/refer`` checkout is retained in ``datasets`` for
provenance, but its released ``refer.py`` is Python 2 source.  This loader
reads the unchanged official pickle/instances files and exposes the subset of
the REFER API required by the manifest builder.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Sequence


class REFER:
    """Load official RefCOCOg UMD data with the conventional REFER methods."""

    def __init__(self, data_root: str, dataset: str = "refcocog", splitBy: str = "umd") -> None:
        if dataset != "refcocog" or splitBy != "umd":
            raise ValueError("Routed Grounding Repair only permits RefCOCOg with the UMD split")
        dataset_dir = Path(data_root).expanduser().resolve() / dataset
        refs_path = dataset_dir / "refs(umd).p"
        instances_path = dataset_dir / "instances.json"
        if not refs_path.is_file() or not instances_path.is_file():
            raise FileNotFoundError(f"Missing official REFER files under {dataset_dir}")

        with refs_path.open("rb") as handle:
            self.refs = pickle.load(handle, encoding="latin1")
        with instances_path.open("r", encoding="utf-8") as handle:
            instances = json.load(handle)

        self.Refs = {ref["ref_id"]: ref for ref in self.refs}
        self.Anns = {ann["id"]: ann for ann in instances["annotations"]}
        self.Imgs = {image["id"]: image for image in instances["images"]}
        self.Cats = {category["id"]: category["name"] for category in instances["categories"]}
        self.imgToRefs: dict[Any, list[dict[str, Any]]] = {}
        self.imgToAnns: dict[Any, list[dict[str, Any]]] = {}
        for ref in self.refs:
            self.imgToRefs.setdefault(ref["image_id"], []).append(ref)
        for ann in instances["annotations"]:
            self.imgToAnns.setdefault(ann["image_id"], []).append(ann)

    @staticmethod
    def _many(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list | tuple | set):
            return list(value)
        return [value]

    def getRefIds(
        self,
        image_ids: Sequence[Any] | Any = (),
        cat_ids: Sequence[Any] | Any = (),
        ref_ids: Sequence[Any] | Any = (),
        split: str = "",
    ) -> list[Any]:
        refs = list(self.refs)
        image_ids = set(self._many(image_ids))
        cat_ids = set(self._many(cat_ids))
        ref_ids = set(self._many(ref_ids))
        if image_ids:
            refs = [ref for ref in refs if ref["image_id"] in image_ids]
        if cat_ids:
            refs = [ref for ref in refs if ref["category_id"] in cat_ids]
        if ref_ids:
            refs = [ref for ref in refs if ref["ref_id"] in ref_ids]
        if split:
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Unsupported RefCOCOg UMD split: {split}")
            refs = [ref for ref in refs if ref.get("split") == split]
        return [ref["ref_id"] for ref in refs]

    def loadRefs(self, ref_ids: Sequence[Any] | Any = ()) -> list[dict[str, Any]]:
        ids = self._many(ref_ids)
        return [self.Refs[ref_id] for ref_id in ids] if ids else list(self.refs)

    def loadAnns(self, ann_ids: Sequence[Any] | Any = ()) -> list[dict[str, Any]]:
        ids = self._many(ann_ids)
        return [self.Anns[ann_id] for ann_id in ids]

    def loadImgs(self, image_ids: Sequence[Any] | Any = ()) -> list[dict[str, Any]]:
        ids = self._many(image_ids)
        return [self.Imgs[image_id] for image_id in ids]
