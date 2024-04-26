from flask import Flask, render_template, request, redirect, url_for
import torch
from PIL import Image
import torchvision.transforms as transforms
from Transformers import ViT

app = Flask(__name__)

# Load your pre-trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ViT(num_classes=10)  # Adjust parameters as necessary
model.load_state_dict(torch.load('model.pth'))  # Load the model weights
model.to(device)
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((48, 48)),  # Adjust size as per your model's training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            image = Image.open(file.stream).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                prediction = model(image)
                predicted_class = torch.argmax(prediction, dim=1).item()
            return render_template('result.html', class_id=predicted_class)
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
