# Copyright 2022 IBM All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Data types and functions for working with class labels and synset files.
"""

from typing import Any, Dict, List, NamedTuple, Set


class Label(NamedTuple):
    """The class labels for the RestNet model have a synset id, as well as
    several names for each label."""

    synset_id: str
    names: Set[str]

    @classmethod
    def from_str(cls, s: str) -> "Label":
        """Loads the label from a line in the syset file

         A sample line might look like:
        "n01531178 goldfinch, Carduelis carduelis
        """
        return Label(synset_id=s[0:9], names=set(n.strip() for n in s[9:].split(",")))

    def dict(self) -> Dict[Any, Any]:
        """Returns the label as a dict"""
        return {"synset_id": self.synset_id, "names": list(self.names)}


def get_class_labels(label_file: str) -> List[Label]:
    """Returns the class labels that are associated with model outputs

    The order of the labels matches the order of the outputs of the
    ResNet model.
    The outputs of the model are described here:
    https://github.com/onnx/models/tree/main/vision/classification/resnet#output
    """
    with open(label_file, "r") as f:
        return [Label.from_str(l) for l in f]
