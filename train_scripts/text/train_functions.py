def train_one_epoch(model, train_dataloader, optimizer, accelerator, device):
    model.train()
    total_loss = 0
    iter_counter = 0

    for input_ids, attention_masks, labels in train_dataloader:
        # optimizer.zero_grad()
        # outputs = model(input_ids=input_ids.to(device),
        #                 attention_mask=attention_masks.to(device),
        #                 labels=labels.to(device))
        # loss = outputs.loss
        # loss.backward()
        # optimizer.step()

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_masks,
                        labels=labels)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()

        total_loss += loss.item()
        iter_counter += 1

    total_loss /= iter_counter
    return total_loss
