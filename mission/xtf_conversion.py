import numpy as np
from PIL import Image
from pyxtf import xtf_read, concatenate_channel, XTFHeaderType
from pathlib import Path

def convert_xtf_to_png(    xtf_path: Path,    output_dir: Path,) -> Path:

    xtf_path = Path(xtf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{xtf_path.stem}.png"
    (fh, p) = xtf_read(xtf_path)
    
    if XTFHeaderType.sonar in p:
        upper_limit = 2 ** 16


        np_chan = concatenate_channel(p[XTFHeaderType.sonar], file_header=fh, channel=0, weighted=False)
    

        np_chan.clip(0, upper_limit - 1, out=np_chan)
    

        np_chan = np.log10(np_chan + 1, dtype=np.float32)


        vmin = np_chan.min()
        vmax = np_chan.max()

  
        np_chan_8bit = ((np_chan - vmin) / (vmax - vmin)) * 255
        np_chan_8bit = np.clip(np_chan_8bit, 0, 255)


        img = Image.fromarray(np_chan_8bit.astype(np.uint8))
        img = img.resize((int(img.size[0]/2), img.size[1]), Image.Resampling.LANCZOS)
        img.save(output_path)

        return output_path
