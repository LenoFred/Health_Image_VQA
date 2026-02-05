import sys, os
import torch
from PIL import Image
from torchvision import transforms
from preprocessing import preprocess_image
from utils import UNetPlusPlus, Tokenizer
from vqa_module import VQAModule

# 1) Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device}")

# 2) Load segmentation model (quantized or not)
seg = UNetPlusPlus(
    backbone='mobilenet_v2',
    pretrained=True,
    in_channels=3,
    classes=1
).to(device)

seg_ckpt = os.path.join('models', 'segmentation', 'best_segmentation_quantized.pt')
seg.load_state_dict(torch.load(seg_ckpt, map_location=device))
seg.eval()

# 3) Load tokenizer and VQA module
tok = Tokenizer.load('code/src/utils_vocab.json')

vqa = VQAModule(
    vision_backbone=seg,
    tokenizer=tok
).to(device)

vqa_ckpt = os.path.join('models', 'vqa', 'best_vqa.pt')
vqa.load_state_dict(torch.load(vqa_ckpt, map_location=device))
vqa.eval()

# 4) Preprocessing transform
tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.225,0.225]),
])

def run_demo(img_path: str, question: str = None, area_thresh: float = 0.02):
    # --- Preprocess image
    img_np = preprocess_image(img_path)
    img = tf(Image.fromarray(img_np).convert('RGB')).to(device)

    # --- Segmentation
    with torch.no_grad():
        mask_logits = seg(img.unsqueeze(0))        # [1,1,H,W]
        mask = (torch.sigmoid(mask_logits) > 0.5).float()

    # --- Binary “disease found” fallback via area threshold
    area_pct = mask.sum() / mask.numel()
    flag = 'YES' if area_pct.item() > area_thresh else 'NO'
    print(f">>> Disease Found: {flag}  (seg area = {area_pct.item():.3f})")

    # --- Optional VQA
    if question:
        # pass raw question string in a list
        with torch.no_grad():
            ans_score = vqa(
                img.unsqueeze(0),  # [1,3,224,224]
                mask,              # [1,1,224,224]
                [question]         # list[str] length=1
            )
        ans = 'YES' if ans_score.item() > 0.5 else 'NO'
        print(f">>> Q: {question}")
        print(f">>> A: {ans}  (score = {ans_score.item():.3f})")

    # return the mask for optional overlays
    return mask.squeeze(0).cpu().numpy()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path> [optional question]")
        sys.exit(1)

    img_file = sys.argv[1]
    q = sys.argv[2] if len(sys.argv) > 2 else None
    run_demo(img_file, q)
