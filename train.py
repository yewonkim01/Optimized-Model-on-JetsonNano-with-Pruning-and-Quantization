import torch.nn as nn
import torch.optim as optim

def original_train(model, device, data):
    epochs = 5
    learning_rate = 1e-3
    total_samples = 0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for e in range(epochs):
        total_loss = 0

        for idx, (images, labels) in enumerate(data):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_samples += labels.size(0)
            total_loss += loss.item()

            # if idx % 100 == 0:
            # print(f"idx {idx}, Runnning loss: {total_loss / len(train_loader)}")
        # print(f"epoch {e} : total_loss : {total_loss}")

    return model