import os

from flask import Flask, render_template, request, redirect, url_for
import torch
from PIL import Image
from torchvision import transforms, datasets
from torchvision.models import vit_l_16
from werkzeug.utils import secure_filename

from Transformers import ViT

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = vit_l_16(pretrained=False)
# num_features = model.heads.head.in_features
# model.heads.head = torch.nn.Linear(num_features, 38)
model = torch.load('my-model_full.pth')
model.to(device)
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(
    '../New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train', transform=transform)
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def get_class_name(class_id):
    return idx_to_class.get(class_id, "Unknown")


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        create_directory(app.config['UPLOAD_FOLDER'])

        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image = Image.open(filepath).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                prediction = model(image)
                predicted_class_idx = torch.argmax(prediction, dim=1).item()
                predicted_class = get_class_name(predicted_class_idx)
            return render_template('result.html', filename=filename, class_name=predicted_class)

    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
