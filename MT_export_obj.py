

"""
Usage:
python MT_export_obj.py 
       --model_dir <your_pretrained_model_dir> 
       --audio_file <your_speech_snippet.m4a> 
"""
import os
import argparse
import numpy as np
import torch as th
from pathlib import Path
from pydub import AudioSegment
from moviepy.editor import *

from utils.renderer import Renderer
from utils.helpers import smooth_geom, load_mask, get_template_verts, load_audio, audio_chunking
from models.vertex_unet import VertexUnet
from models.context_model import ContextModel
from models.encoders import MultimodalEncoder


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",
                    type=str,
                    default="pretrained_models",
                    help="directory containing the models to load")
parser.add_argument("--audio_file",
                    type=str,
                    default="/is/cluster/fast/scratch/rdanecek/testing/enspark/baselines/lrs3_audios_only",
                    help="wave file to use for face animation"
                    )
parser.add_argument("--out_path",
                    type=str,
                    default="//is/cluster/fast/scratch/rdanecek/testing/enspark/baselines/MT_lrs3_test/sub1",
                    help="wave file to use for face animation"
                    )
parser.add_argument("--face_template",
                    type=str,
                    default="assets/face_template.obj",
                    help=".obj file containing neutral template mesh"
                    )
args = parser.parse_args()

# set up
template_verts = get_template_verts(args.face_template)
mean = th.from_numpy(np.load("assets/face_mean.npy"))
stddev = th.from_numpy(np.load("assets/face_std.npy"))
forehead_mask = th.from_numpy(load_mask("assets/forehead_mask.txt", dtype=np.float32)).cuda()
neck_mask = th.from_numpy(load_mask("assets/neck_mask.txt", dtype=np.float32)).cuda()
renderer = Renderer("assets/face_template.obj")
geom_unet = VertexUnet(classes=128,
                       heads=16,
                       n_vertices=6172,
                       mean=mean,
                       stddev=stddev,
                       )
geom_unet.load(args.model_dir)
geom_unet.cuda().eval()
context_model = ContextModel(classes=128,
                             heads=16,
                             audio_dim=128
                             )
context_model.load(args.model_dir)
context_model.cuda().eval()
encoder = MultimodalEncoder(classes=128,
                            heads=16,
                            expression_dim=128,
                            audio_dim=128,
                            n_vertices=6172,
                            mean=mean,
                            stddev=stddev,
                            )
encoder.load(args.model_dir)
encoder.cuda().eval()
j = 0
for i, p in enumerate(Path(args.audio_file).glob('*')):
    sample_dir = p.stem
    out_path = os.path.join(args.out_path, sample_dir)
    os.makedirs(out_path, exist_ok=True)
    print(f"Processing {sample_dir} ({i+1}/{len(list(Path(args.audio_file).glob('*')))})")
    audio_fname = os.path.join(args.audio_file, sample_dir)

    for audio_file in Path(audio_fname).glob("*.wav"):
        
        test_name = audio_file.stem
        obj_path = os.path.join(out_path, test_name, "meshes")
        
        
        if not Path(obj_path).exists():
            print(f"Resuming audio_file: {test_name}, j: {j}")
            j += 1
            os.makedirs(obj_path, exist_ok=True)
            
            # videoclip = VideoFileClip(str(audio_file)) # lrs3 are wav video files
            # audioclip = videoclip.audio
            # audio_dir = Path(out_path) / test_name/ "audio"
            # audio_dir.mkdir(exist_ok=True, parents=True)
            # wav_path = os.path.join(audio_dir, "audio.wav")
            # audioclip.write_audiofile(wav_path)
            
            wav_path = str(audio_file)
            
            # name = audio_file.stem
            # track = AudioSegment.from_file(audio_file, format="m4a")
            # wav = track.export(f"{Path(args.audio_file) / name}.wav", format="wav")
            
            audio = load_audio(wav_path)
            audio = audio_chunking(audio, frame_rate=30, chunk_size=16000)
            with th.no_grad():
                audio_enc = encoder.audio_encoder(audio.cuda().unsqueeze(0))["code"]
                one_hot = context_model.sample(audio_enc, argmax=False)["one_hot"]
                T = one_hot.shape[1]
                geom = template_verts.cuda().view(1, 1, 6172, 3).expand(-1, T, -1, -1).contiguous()
                result = geom_unet(geom, one_hot)["geom"].squeeze(0)
            result = smooth_geom(result, forehead_mask)
            result = smooth_geom(result, neck_mask)
            # obj_dir = Path(args.audio_file) / "MT" /f"{name}_objs"
            # obj_dir.mkdir(exist_ok=True, parents=True)
            # renderer.to_obj(result, obj_dir, name)
            renderer.to_obj(result, obj_path)

# for wav in Path(args.audio_file).glob("*.wav"): wav.unlink()
print("Changing permissions of the output files")
os.system("find /is/cluster/fast/scratch/rdanecek/testing/enspark/baselines -type d -exec chmod 755 {} +")
print("Done")
