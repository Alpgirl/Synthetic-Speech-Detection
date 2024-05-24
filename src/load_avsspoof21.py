import datasets
import os

_HOMEPAGE = "https://www.asvspoof.org/index2021.html"
_URLS = {
    "partition_0": "https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part00.tar.gz?download=1",
    "partition_1": "https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part01.tar.gz?download=1",
    "partition_2": "https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part02.tar.gz?download=1",
    "partition_3": "https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part03.tar.gz?download=1",
    "metadata": "https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz",
}
_CITATION = """\

"""

_DESCRIPTION = """\

"""

class ASVspoof2021(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG = datasets.BuilderConfig(name="DF", description="DeepFake (DF)")
    
    def __init__(self):
        super().__init__

    def _info(self):
        print(self.BUILDER_CONFIG.name)
        features = datasets.Features(
            {
                "audio_file_name": datasets.Value("string"),
                "audio": datasets.Audio(sampling_rate=16_000),
                "system_id": datasets.Value("string"),
                "src": datasets.Value("string"),
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
        urls_to_download = _URLS
        data_dir = dl_manager.download_and_extract(urls_to_download)
        
        full_data_dir = os.path.join(dl_manager.manual_dir, 'ASVspoof2021-DF-full')
        os.makedirs(full_data_dir, exist_ok=True)

        audio_dir = os.path.join(full_data_dir, 'flac')
        key_dir = os.path.join(full_data_dir, 'keys')
        
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(key_dir, exist_ok=True)
        
        for i in range(4):
            partition_dir = os.path.join(data_dir[f"partition_{i}"], 'ASVspoof2021_DF_eval', 'flac')
            for audio_filename in os.listdir(partition_dir):
                src_path = os.path.join(partition_dir, audio_filename)
                dst_path = os.path.join(audio_dir, audio_filename)
                if not os.path.isfile(dst_path):
                    os.symlink(src_path, dst_path)
        
        metadata_file_path = os.path.join(data_dir["metadata"], 'keys', 'DF', 'CM', 'trial_metadata.txt')
        dst_path = os.path.join(key_dir, 'keys.txt')
        if not os.path.islink(dst_path):
            os.symlink(metadata_file_path, dst_path)
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "metadata_filepath": os.path.join(key_dir, 'keys.txt'),
                    "audios_dir": audio_dir
                },
            )       
        ]
    
    def _generate_examples(self, metadata_filepath, audios_dir):
        data = []
        with open(metadata_filepath) as f:
            for i, line in enumerate(f.readlines()):
                fields = line.strip().split()
                result = {
                    "speaker_id": fields[0],
                    "audio_file_name": fields[1],
                    "system_id": fields[4],
                    "src": fields[3],
                    "key": fields[5],
                }
                result["audio"] = os.path.join(audios_dir, result['audio_file_name'] + ".flac")
                data.append(result)
        return data
