import datasets
import os

_HOMEPAGE = "https://datashare.ed.ac.uk/handle/10283/3336"
_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip"
_CITATION = """\
@InProceedings{Todisco2019,
  Title     = {{ASV}spoof 2019: {F}uture {H}orizons in {S}poofed and {F}ake {A}udio {D}etection},
  Author    = {Todisco, Massimiliano and
               Wang, Xin and
               Sahidullah, Md and
               Delgado, H ́ector and
               Nautsch, Andreas and
               Yamagishi, Junichi and
               Evans, Nicholas and
               Kinnunen, Tomi and
               Lee, Kong Aik},
  booktitle = {Proc. of Interspeech 2019},
  Year      = {2019}
}
"""

_DESCRIPTION = """\
This is a database used for the Third Automatic Speaker Verification Spoofing
and Countermeasuers Challenge, for short, ASVspoof 2019 (http://www.asvspoof.org)
organized by Junichi Yamagishi, Massimiliano Todisco, Md Sahidullah, Héctor
Delgado, Xin Wang, Nicholas Evans, Tomi Kinnunen, Kong Aik Lee, Ville Vestman,
and Andreas Nautsch in 2019.
"""

class ASVspoof2019(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG = datasets.BuilderConfig(name="LA", description="Logical access (LA)")
    
    def __init__(self):
        super().__init__

    def _info(self):
        # which features to expect in dataset
        print(self.BUILDER_CONFIG.name)
        features = datasets.Features(
            {
                "speaker_id": datasets.Value("string"),
                "audio_file_name": datasets.Value("string"),
                "audio": datasets.Audio(sampling_rate=16_000),
                "system_id": datasets.Value("string"),
                "key": datasets.ClassLabel(names=["bonafide", "spoof"]),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=("audio", "key"),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "metadata_filepath": os.path.join(
                        data_dir,
                        self.BUILDER_CONFIG.name,
                        f"ASVspoof2019_{self.BUILDER_CONFIG.name}_cm_protocols",
                        f"ASVspoof2019.{self.BUILDER_CONFIG.name}.cm.train.trn.txt",
                    ),
                    "audios_dir": os.path.join(
                        data_dir, self.BUILDER_CONFIG.name, f"ASVspoof2019_{self.BUILDER_CONFIG.name}_train", "flac"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "metadata_filepath": os.path.join(
                        data_dir,
                        self.BUILDER_CONFIG.name,
                        f"ASVspoof2019_{self.BUILDER_CONFIG.name}_cm_protocols",
                        f"ASVspoof2019.{self.BUILDER_CONFIG.name}.cm.dev.trl.txt",
                    ),
                    "audios_dir": os.path.join(
                        data_dir, self.BUILDER_CONFIG.name, f"ASVspoof2019_{self.BUILDER_CONFIG.name}_dev", "flac"
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "metadata_filepath": os.path.join(
                        data_dir,
                        self.BUILDER_CONFIG.name,
                        f"ASVspoof2019_{self.BUILDER_CONFIG.name}_cm_protocols",
                        f"ASVspoof2019.{self.BUILDER_CONFIG.name}.cm.eval.trl.txt",
                    ),
                    "audios_dir": os.path.join(
                        data_dir, self.BUILDER_CONFIG.name, f"ASVspoof2019_{self.BUILDER_CONFIG.name}_eval", "flac"
                    ),
                },
            ),
        ]
    
    def _generate_examples(self, metadata_filepath, audios_dir):
        data = []
        with open(metadata_filepath) as f:
            for i, line in enumerate(f.readlines()):
                speaker_id, audio_file_name, _, system_id, key = line.strip().split()
                result = {
                    "speaker_id": speaker_id,
                    "audio_file_name": audio_file_name,
                    "system_id": system_id,
                    "key": key,
                }
                result["audio"] = os.path.join(audios_dir, audio_file_name + ".flac")
                data.append(result)
        return data