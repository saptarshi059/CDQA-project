# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""covid_qa_cleaned_CS: Connor Heaton/Saptarshi Sengupta"""

import datasets
import requests
import json
import os

logger = datasets.logging.get_logger(__name__)


# You can copy an official description
_DESCRIPTION = """\
Cleaned version of COVID-QA containing fixes as mentioned in ``Towards Efficient Methods in Medical Question Answering using Knowledge Graph Embeddings``.
"""


_CITATION = """\
@inproceedings{sengupta2024towards,
  title={Towards Efficient Methods in Medical Question Answering using Knowledge Graph Embeddings},
  author={Sengupta, Saptarshi and Heaton, Connor and Cui, Suhan and Sarkar, Soumalya and Mitra, Prasenjit},
  booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={5089--5096},
  year={2024},
  organization={IEEE}
}
"""

_HOMEPAGE = "https://ieeexplore.ieee.org/abstract/document/10821824"


_LICENSE = "Apache License 2.0"


_URL = "https://github.com/saptarshi059/CDQA-v2-Auxilliary-Loss/tree/main/data/covid_qa_cleaned_CS"
_URLs = {"covid_qa_cleaned_CS": _URL + "covid_qa_cleaned_CS.json"}


class CovidQADeepsetCleaned(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="covid_qa_cleaned_CS", version=VERSION, description="Cleaned version of COVID-QA (deepset) by Connor Heaton & Saptarshi Sengupta"),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "document_id": datasets.Value("int32"),
                "context": datasets.Value("string"),
                "question": datasets.Value("string"),
                "is_impossible": datasets.Value("bool"),
                "id": datasets.Value("int32"),
                "answers": datasets.features.Sequence(
                    {
                        "text": datasets.Value("string"),
                        "answer_start": datasets.Value("int32"),
                    }
                ),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            license=_LICENSE,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        url = _URLs[self.config.name]
        downloaded_filepath = dl_manager.download_and_extract(url)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_filepath},
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            covid_qa = json.load(f)
            for article in covid_qa["data"]:
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"].strip()
                    document_id = paragraph["document_id"]
                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        is_impossible = qa["is_impossible"]
                        id_ = qa["id"]

                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        answers = [answer["text"].strip() for answer in qa["answers"]]

                        # Features currently used are "context", "question", and "answers".
                        # Others are extracted here for the ease of future expansions.
                        yield id_, {
                            "document_id": document_id,
                            "context": context,
                            "question": question,
                            "is_impossible": is_impossible,
                            "id": id_,
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }
