# 🏀 About HoopsTalk
<img src="docs/images/tsf-gpt2-archi.png" alt="HoopsTalk Logo" width="500"/>
HoopsTalk harnesses the power of generative AI to provide engaging and ensuring a comprehensive and immersive experience for fans.
Our proposed caption generating architecture consists of (i) TimeSformer Encoder and (ii) Transformer-based Decoder with multi-head self attention.

## Webscraper
1. Get the data from https://www.dropbox.com/sh/x3zpttp7bjevb3r/AAAeFLnIeBMBXa9DNQD4a8TOa?e=2&dl=0 and put the content inside `data/raw/NSVA_Data/NSVA_Data`.
2. Run the `webscraper.py`

## Text-to-Speech
1. Download the models from https://huggingface.co/enlyth/baj-tts/tree/main/models and put it inside the `models` directory
2. Install the requirements, `pip install TTS==0.22.0` or just run `pip install -r requirements.txt`.
3. Run `src/tts.py` from the root directory.
4. Check the generated `.wav` file in `output` directory.
