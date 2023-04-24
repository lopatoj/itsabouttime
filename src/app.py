import einops
import torch
import cv2
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    verbose = args.verbose
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # DATASET
  images = [x for x in natsorted(os.listdir(args.dir)) if ('.jpg' in x) or ('.png' in x)]

  # MODEL
  model_stn = models.resnet50()
  model_stn.fc = nn.Linear(2048, 8)
  model = models.resnet50()
  model.fc = nn.Linear(2048, 720)
  resume_path = './models/{}.pth'.format(verbose)
  stn_resume_path = './models/{}_st.pth'.format(verbose)
  model.load_state_dict(torch.load(resume_path, map_location=torch.device('cpu')))
  model_stn.load_state_dict(torch.load(stn_resume_path, map_location=torch.device('cpu')))
  model_stn.to(device)
  model.to(device)

  for count in range(len(images)):
    with torch.no_grad():
      model.eval()
      model_stn.eval()

      #MODEL
      img = cv2.imread(os.path.join(args.dir, images[count]))
      img = cv2.resize(img, (224, 224))/255.
      img = einops.rearrange(img, 'h w c -> c h w')
      img = torch.Tensor(img)
      img = img.float().to(device)
      img = torch.unsqueeze(img, 0)

      pred_st = model_stn(img)
      pred_st = torch.cat([pred_st,torch.ones(1,1).to(device)], 1)
      Minv_pred = torch.reshape(pred_st, (-1, 3, 3))
      img_ = warp(img, Minv_pred)
      pred = model(img_)

      #top 3 predictions
      max_pred = torch.argsort(pred, dim=1, descending=True)
      max_pred = max_pred[0,:3]
      max_h = max_pred[0] // 60
      max_m = max_pred[0] % 60

      print(f"[{images[count]}] is roughly {max_h.cpu().numpy()}:{max_m.cpu().numpy():0>2}.\n")