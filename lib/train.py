import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from lib.utils import Logger, plot_grad_flow
from lib.evaluate import get_sym_emb, accuracy

def train(word_emb, vision_model, language_model, loss_model,
          train_loader, val_loader, word_dict, ent_dict, pred_dict,
          n_epochs, val_freq, out_dir, cfg):

    os.makedirs(out_dir, exist_ok=True)
    params = list(vision_model.parameters()) + list(language_model.parameters())
    params = [ param for param in params if param.requires_grad ]
    named_params = list(vision_model.named_parameters()) + list(language_model.named_parameters())
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

            image_ids = data[0]
            if len(image_ids) < cfg.train.batch_size: continue
            images = data[1].float().cuda()
            sbj_boxes = data[2].float().cuda()
            obj_boxes = data[3].float().cuda()
            rel_boxes = data[4].float().cuda()
            sbj_tokens = data[5].cuda()
            obj_tokens = data[6].cuda()
            rel_tokens = data[7].cuda()
            sbj_seq_lens = data[8].long().cuda()
            obj_seq_lens = data[9].long().cuda()
            rel_seq_lens = data[10].long().cuda()

            tic_2 = time.time()

            optimizer.zero_grad()

            sbj_t_emb = language_model(word_emb(sbj_tokens), sbj_seq_lens)
            obj_t_emb = language_model(word_emb(obj_tokens), obj_seq_lens)
            rel_t_emb = language_model(word_emb(rel_tokens), rel_seq_lens)
            sbj_v_emb, obj_v_emb, rel_v_emb = vision_model(images, sbj_boxes, obj_boxes, rel_boxes)

            sbj_loss = loss_model(sbj_v_emb, sbj_t_emb)
            obj_loss = loss_model(sbj_v_emb, obj_t_emb)
            rel_loss = loss_model(sbj_v_emb, rel_t_emb)

            loss = sbj_loss + obj_loss + rel_loss

            tic_3 = time.time()

            loss.backward()
            optimizer.step()

            tic_4 = time.time()

            if i % 100 == 0:
                plot_grad_flow(named_params)

            epoch_loss += loss.data.item() * train_loader.batch_size

            logstr = "epoch %2d batch %4d/%d4 | loss %5.2f | %4dms | ^ %4dms | => %4dms" % \
                     (epoch+1, i+1, n_batches, loss.data.item(),
                      1000*(tic_4-tic_0), 1000*(tic_2-tic_0), 1000*(tic_4-tic_2))
            print("%-80s" % logstr, end="\r")

            tic_0 = time.time()

        epoch_loss /= n_batches * train_loader.batch_size

        logstr = "epoch %2d | train_loss: %5.2f" % (epoch+1, epoch_loss)

        if (epoch + 1) % val_freq == 0:
            vision_model.train(False)
            vision_model.eval()
            language_model.train(False)
            language_model.eval()
            ent_acc, rel_acc = validate(word_emb, vision_model, language_model, val_loader,
                                        word_dict, ent_dict, pred_dict, cfg.language_model.tokens_length)
            logstr += " ent_acc: %.3f rel_acc: %.3f" % (ent_acc, rel_acc)
            vision_model.train(True)
            language_model.train(True)

        print("%-80s" % logstr)
        logger.write("%-80s" % logstr)

        vision_model_path = os.path.join(out_dir, "vision_model_%d.pth" % (epoch+1))
        torch.save(vision_model.state_dict(), vision_model_path)
        language_model_path = os.path.join(out_dir, "language_model_%d.pth" % (epoch + 1))
        torch.save(language_model.state_dict(), language_model_path)


def validate(word_emb, vision_model, language_model, loader,
             word_dict, ent_dict, pred_dict, tokens_length):

    ent_embs = get_sym_emb(word_emb, language_model, word_dict, ent_dict, tokens_length)
    pred_embs = get_sym_emb(word_emb, language_model, word_dict, pred_dict, tokens_length)

    ent_acc, rel_acc = accuracy(vision_model, loader, ent_embs, pred_embs)

    return ent_acc, rel_acc