from pydub.utils import mediainfo

def extract_flac_info(file_path):
    metadata = mediainfo(file_path)
    return metadata

# Usage
flac_file_path = "LibriSpeech/train-clean-100/19/198/19-198-0000.flac"
info = extract_flac_info(flac_file_path)
print(info)
