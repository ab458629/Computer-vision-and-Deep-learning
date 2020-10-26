# Training model
for epoch in range(num_epoches):
        print('*' * 25, 'epoch {}'.format(epoch + 1), '*' * 25)
        running_loss = 0.0
        running_acc = 0.0
        for i, data in tqdm(enumerate(train_loader, 1)):
            img, label = data
            # cuda
            if use_gpu:
                img = img.cuda()
                label = label.cuda()
            img = Variable(img)
            label = Variable(label)
            # Forward propagation
            out = model(img)
            loss = criterion(out, label)
            running_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            accuracy = (pred == label).float().mean()
            running_acc += num_correct.item()
            # Back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(running_loss)
            train_accuracy.append(running_acc)
        print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(train_dataset))))

        model.eval()
        eval_loss = 0
        eval_acc = 0
        for data in test_loader:
            img, label = data
            if use_gpu:
                img = Variable(img, volatile=True).cuda()
                label = Variable(label, volatile=True).cuda()
            else:
                img = Variable(img, volatile=True)
                label = Variable(label, volatile=True)
            out = model(img)
            loss = criterion(out, label)
            eval_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            eval_acc += num_correct.item()

            val_loss.append(eval_loss)
            val_accuracy.append(eval_acc)
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
            test_dataset)), eval_acc / (len(test_dataset))))

    # Save model
    torch.save(model.state_dict(), './vgg16.pth')
