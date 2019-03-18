import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from lib.utils import Logger
from lib.evaluate import get_sym_emb, accuracy

def train(vision_model, language_model, loss_model,
          train_loader, val_loader,
          word_dict, ent_dict, pred_dict,
          n_epochs, val_freq, out_dir, cfg):

    os.makedirs(out_dir, exist_ok=True)
    params = list(vision_model.parameters()) + list(language_model.parameters())
    params = [ param for param in params if param.requires_grad ]
    optimizer = torch.optim.Adam(params, lr=cfg.train.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1,
                                                gamma=cfg.train.learning_rate_decay)
    logger = Logger(os.path.join(out_dir, "log.txt"))
    n_batches = len(train_loader)

    for epoch in range(n_epochs):

        scheduler.step()
        epoch_loss = 0.0
        tic_0 = time.time()

        for i, data in enumerate(train_loader):

            tic_1 = time.time()

            images = data[1].float().cuda()
            sbj_boxes = data[2].float().cuda()
            obj_boxes = data[3].float().cuda()
            rel_boxes = data[4].float().cuda()
            sbj_tokens = data[5].cuda()
            obj_tokens = data[6].cuda()
            rel_tokens = data[7].cuda()

            tic_2 = time.time()

            optimizer.zero_grad()

            sbj_v_emb, obj_v_emb, rel_v_emb = vision_model(images, sbj_boxes, obj_boxes, rel_boxes)
            sbj_t_emb, obj_t_emb, rel_t_emb = language_model(sbj_tokens, obj_tokens, rel_tokens)

            sbj_loss = loss_model(sbj_v_emb, sbj_t_emb)
            obj_loss = loss_model(obj_v_emb, obj_t_emb)
            rel_loss = loss_model(rel_v_emb, rel_t_emb)

            loss = sbj_loss + obj_loss + rel_loss

            tic_3 = time.time()

            loss.backward()
            optimizer.step()

            tic_4 = time.time()

            epoch_loss += loss.data.item() * train_loader.batch_size

            logstr = "epoch %2d batch %4d/%d4 | ^ %4dms | => %4dms | <= %4dms" % \
                     (epoch+1, i+1, n_batches, 1000*(tic_2-tic_0), 1000*(tic_3-tic_2), 1000*(tic_4-tic_3))
            print("%-80s" % logstr, end="\r")

            tic_0 = time.time()

        epoch_loss /= n_batches * train_loader.batch_size

        logstr = "epoch %2d | train_loss: %5.2f" % (epoch+1, epoch_loss)

        if (epoch + 1) % val_freq == 0:
            vision_model.train(False)
            language_model.train(False)
            ent_acc, rel_acc = validate(vision_model, language_model, val_loader, word_dict, ent_dict, pred_dict, cfg)
            logstr += " ent_acc: %.3f rel_acc: %.3f" % (ent_acc, rel_acc)

        print("%-80s" % logstr)
        logger.write("%-80s" % logstr)

        vision_model_path = os.path.join(out_dir, "vision_model_%d.pth" % (epoch+1))
        torch.save(vision_model.state_dict(), vision_model_path)
        language_model_path = os.path.join(out_dir, "language_model_%d.pth" % (epoch + 1))
        torch.save(language_model.state_dict(), language_model_path)


def validate(vision_model, language_model, loader,
             word_dict, ent_dict, pred_dict, cfg):

    ent_embs = get_sym_emb(language_model, word_dict, ent_dict, cfg.language_model.tokens_length)
    pred_embs = get_sym_emb(language_model, word_dict, pred_dict, cfg.language_model.tokens_length)

    ent_acc, rel_acc = accuracy(vision_model, loader, ent_embs, pred_embs)

    return ent_acc, rel_acc