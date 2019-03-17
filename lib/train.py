import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from lib.utils import Logger

def train(vision_model, language_model, loss_model,
          cfgs, train_loader, val_loader,
          n_epochs, val_freq, out_dir):

    os.makedirs(out_dir, exist_ok=True)
    optim = torch.optim.Adam([vision_model.parameters(), language_model.parameters()])
    logger = Logger(os.path.join(out_dir, "log.txt"))

    epoch_loss = 0.0

    for epoch in range(n_epochs):

        n_batches = len(train_loader)
        tic_0 = time.time()

        for i, data in enumerate(train_loader):

            tic_1 = time.time()

            with torch.cuda.device(0):
                data = [ item.cuda() for item in data ]

            image_ids, images, \
            sbj_boxes, obj_boxes, rel_boxes, \
            sbj_tokens, obj_tokens, rel_tokens = data

            tic_2 = time.time()

            optim.zero_grad()

            sbj_v_emb, obj_v_emb, rel_v_emb = vision_model(images, sbj_boxes, obj_boxes, rel_boxes)
            sbj_t_emb, obj_t_emb, rel_t_emb = language_model(sbj_tokens, obj_tokens, rel_tokens)

            sbj_loss = loss_model(sbj_v_emb, sbj_t_emb)
            obj_loss = loss_model(obj_v_emb, obj_t_emb)
            rel_loss = loss_model(rel_v_emb, rel_t_emb)

            loss = sbj_loss + obj_loss + rel_loss

            tic_3 = time.time()

            loss.backward()
            optim.step()

            tic_4 = time.time()

            epoch_loss += loss.data.item() * train_loader.batch_size

            logstr = "epoch %2d batch %4d/%d4 | ^ %4dms | => %4dms | <= %4dms" % \
                     (epoch+1, i+1, n_batches, 1000*(tic_2-tic_0), 1000*(tic_3-tic_2), 1000*(tic_4-tic_3))
            print("%-80s" % logstr, end="\r")

            tic_0 = time.time()

        epoch_loss /= n_batches * train_loader.batch_size

        logstr = "epoch %2d | train_loss: %5.2f" % (epoch+1, epoch_loss)

        # if (epoch + 1) % val_freq == 0:
        #     vision_model.train(False)
        #     language_model.train(False)

        vision_model_path = os.path.join(out_dir, "vision_model_%d.pth" % (epoch+1))
        torch.save(vision_model.state_dict(), vision_model_path)
        language_model_path = os.path.join(out_dir, "language_model_%d.pth" % (epoch + 1))
        torch.save(language_model.state_dict(), language_model_path)
