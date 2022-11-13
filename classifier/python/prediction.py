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
Data type for a prediction
"""
import json
from collections.abc import Iterable
from typing import Any, Dict, NamedTuple

from labels import Label


class Prediction(NamedTuple):
    "Represents a prediction & score"
    label: Label
    score: float

    def dict(self) -> Dict[Any, Any]:
        """Converts the prediction to dict"""
        return {"Label": self.label.dict(), "score": self.score}
