# itsabouttime
Code repository for "It's About Time: Analog clock Reading in the Wild"

Packages required: 
`pytorch` (used 1.9, any reasonable version should work), `kornia` (for homography), `einops`, `scikit-learn` (for RANSAC), `tensorboardX` (for logging)

Using pretrained model:
- prediction `python predict.py` will predict on your data (or by default, what is in `data/demo`)
- evaluation `python eval.py`

Training:
- `sh full_cycle.sh` should do the job

Dataset (Train):
- SynClock is generated on the fly (via `SynClock.py`)
- Timelapse will be uploaded later.

Dataset (Eval)
- COCO and OpenImages: The `.csv` files in `data/` contains the image ids, predicted bbox's (by CBNetV2), gt bbox's, and the manual time label. We will upload this subset later for convenience.
- Clock Movies do not contain bbox's. We may not be able to release the data directly due to copyright, but the csv files do contain the image file names, and they are scraped from https://theclock.fandom.com/wiki/Special:NewFiles