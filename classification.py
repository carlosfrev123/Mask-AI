import numpy as np
import torch
import torchvision.transforms as T
from torchvision import datasets, transforms
from PIL import Image
import os
import json
from tqdm import tqdm
import sklearn.metrics as metrics
import ssl
from roboflow import Roboflow
import os
from dotenv import load_dotenv
import copy


# load_dotenv()

# rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
# project = rf.workspace("taschenbier").project("acne-type-classification")
# version = project.version(3)
# dataset = version.download("folder")

ssl._create_default_https_context = ssl._create_unverified_context

dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

class DinoVisionTransformerClassifier(torch.nn.Module):
    def __init__(self,num_classes=10):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = dinov2_vits14
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(384, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 3)
        )

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x
    
class Utils:
    def __init__(self):
        self.transform_image = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


    def load_images(self, img_paths: list) -> torch.Tensor:
        images = [self.transform_image(Image.open(img).convert('RGB')) for img in img_paths]
        batch = torch.stack(images).to(self.device)
        return batch

    def train(self, model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                    dataloader = train_loader
                else:
                    model.eval()
                    dataloader = val_loader

                running_loss = 0.0
                running_corrects = 0
                for inputs, labels in dataloader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()
                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = running_corrects.double() / len(dataloader.dataset)
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_model_wts)
        return model


    def compute_embeddings(self, files, batch_size=32):
        all_embeddings = {}
        with torch.no_grad():
            for i in tqdm(range(0, len(files), batch_size)):
                batch_files = files[i:i+batch_size]
                embeddings = self.model(self.load_images(batch_files))
                for j, file in enumerate(batch_files):
                    all_embeddings[file] = embeddings[j].cpu().numpy().reshape(-1).tolist()

        with open("all_embeddings.json", "w") as f:
            json.dump(all_embeddings, f)
        return all_embeddings




# # Run the model on the test data
# # Path to test data: Acne-type-classification-2/test, where each subfolder is a label class
# cwd = os.getcwd()
# utils = Utils()
# model = DinoVisionTransformerClassifier()
# test_data = '/content/Acne-type-classification-3/test' #os.path.join(cwd, "Acne-type-classification-2/test")
# train_data = '/content/Acne-type-classification-3/train' #os.path.join(cwd, "Acne-type-classification-2/train")
# train_data = datasets.ImageFolder(train_data, transform=utils.transform_image)
# test_data = datasets.ImageFolder(test_data, transform=utils.transform_image)
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
# val_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.1)

# model.to(utils.device)
# model = utils.train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10)

# # Save the model
# torch.save(model.state_dict(), "model.pth")

# #Load model
# model = DinoVisionTransformerClassifier()
# model.load_state_dict(torch.load("model.pth"))
# model.to(utils.device)
# model.eval()

# # Run the model on a test image and print the prediction
# img = utils.transform_image(Image.open('/content/Acne-type-classification-3/test/1/1.jpg').convert('RGB')).unsqueeze(0).to(utils.device)
# output = model(img)
# _, pred = torch.max(output, 1)
# print(pred.item())