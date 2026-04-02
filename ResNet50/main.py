import torch
import torch.optim as optim
from ResNetFLAME import ResNetFLAME
from UI.flamerenderer import FLAMERenderer
from UI.flamewrapper import FLAMEWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetFLAME(out_features=156).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

flame_wrapper = FLAMEWrapper()
flame_renderer = FLAMERenderer()
# Opcjonalnie: Scheduler (zmniejsza lr, gdy nauka staje w miejscu)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

epochs = 50

def landmark_loss(predicted_params):
    shape = predicted_params[:, :100]
    exp   = predicted_params[:, 100:150]
    pose  = predicted_params[:, 150:156]

    flame_wrapper.shape = shape
    flame_wrapper.expression = exp
    flame_wrapper.pose = pose

    vertices, landmarks, faces = flame_wrapper.generate_mesh()

    flame_renderer.start_view(vertices, landmarks, faces)

    #loss = torch.mean(torch.abs(pred_landmarks - gt_landmarks))
    pass


for epoch in range(epochs):
    model.train()
    total_loss = 0

    for _ in range(10):  # fake batches
        images = torch.randn(8, 3, 224, 224).to(device)
        targets = torch.randn(8, 156).to(device)

        optimizer.zero_grad()

        outputs = model(images)
        landmark_loss(outputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)


        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/10:.4f}")

# for epoch in range(epochs):
#     model.train() # Tryb treningu
#     total_loss = 0
#
#     # for landmarks in train_loader: # images: [B, S, 1, 1]
#     #landmarks = landmarks.to(device)
#
#     # KROK 1: Zerowanie gradientów
#     optimizer.zero_grad()
#
#     # KROK 2: Przewidywanie (Forward)
#     outputs = model(1)
#
#     # KROK 3: Obliczanie błędu (nasz flame_loss z poprzedniej odpowiedzi)
#     # loss = flame_loss(outputs)
#     loss = 0
#     # KROK 4: Wsteczna propagacja (Backward)
#     loss.backward()
#
#     # KROK 5: Aktualizacja wag (Optimizer)
#     optimizer.step()
#
#     total_loss += loss.item()
#
#     # Logowanie postępu i zapis checkpointu
#     # avg_loss = total_loss / len(train_loader)
#     # print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
#
#     # if (epoch + 1) % 10 == 0:
#         #save_checkpoint(model, optimizer, epoch, avg_loss, path=f"models/model_flame_ep{epoch}.pth")
#
# def flame_loss(outputs):
#     print(outputs.shape)
#     p = 0
#
# def train_loader():
#
#     #np.load("../data")
#     p = 0
#
# def save_checkpoint(model, optimizer, epoch, loss, path="checkpoint.pth"):
#     checkpoint = {
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss,
#     }
#     torch.save(checkpoint, path)
#     print(f"Checkpoint zapisany w epoce {epoch}")
#
# def load_checkpoint(path, model, optimizer):
#     checkpoint = torch.load(path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     epoch = checkpoint['epoch']
#     loss = checkpoint['loss']
#     return model, optimizer, epoch, loss