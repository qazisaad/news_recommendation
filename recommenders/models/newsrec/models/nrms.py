# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


# from recommenders.models.newsrec.models.base_model import BaseModel
from recommenders.models.newsrec.models.layers import AdditiveAttention

__all__ = ["NRMSModel"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer



from tqdm import tqdm


class DocEncoder(nn.Module):
    def __init__(self, hparams, weight=None) -> None:
        super(DocEncoder, self).__init__()
        self.hparams = hparams
        if weight is None:
            self.embedding = nn.Embedding(100, 300)
        else:
            self.embedding = nn.Embedding.from_pretrained(weight, freeze=False, padding_idx=0)
            self.embedding.weight = nn.Parameter(self.embedding.weight.float())
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.mha = nn.MultiheadAttention(hparams.word_emb_dim, num_heads=hparams.head_num, dropout=0.1)
        self.proj = nn.Linear(hparams.word_emb_dim, hparams.word_emb_dim)
        self.additive_attn = AdditiveAttention(hparams.word_emb_dim, hparams.attention_hidden_dim)
        self.max_length = hparams.title_size

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print('x', x)
        x = [sent[0] for sent in x]
        inputs = self.tokenizer(x, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # Forward pass
        outputs = self.model(input_ids, attention_mask)

        # Get the embeddings
        embeddings = outputs.last_hidden_state
        # x = x.to(device)
        # embeddings = F.dropout(self.embedding(x), 0.2)
        print('embeddings', embeddings.shape, type(embeddings))
        embeddings = embeddings.permute(1, 0, 2)
        output, _ = self.mha(embeddings, embeddings, embeddings)
        output = output.permute(1, 0, 2)
        output = F.dropout(output)
        output = self.proj(output)
        output, _ = self.additive_attn(output)
        return output
    
class NRMS(nn.Module):
    def __init__(self, hparams, weight=None):
        super(NRMS, self).__init__()
        self.hparams = hparams
        self.doc_encoder = DocEncoder(hparams, weight=weight)
        self.mha = nn.MultiheadAttention(hparams.word_emb_dim, hparams.head_num, dropout=0.1)
        self.proj = nn.Linear(hparams.word_emb_dim, hparams.word_emb_dim)
        self.additive_attn = AdditiveAttention(hparams.word_emb_dim, hparams.attention_hidden_dim)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs):
        """forward

        Args:
            clicks (tensor): [num_user, num_click_docs, seq_len]
            cands (tensor): [num_user, num_candidate_docs, seq_len]
        """
        clicks, cands = inputs
        num_click_docs = clicks.shape[1]
        num_cand_docs = cands.shape[1]
        num_user = clicks.shape[0]
        # seq_len = clicks.shape[2]
        clicks = clicks.reshape(num_click_docs * num_user, -1)
        cands = cands.reshape(num_cand_docs * num_user, -1)
        click_embed = self.doc_encoder(clicks)
        cand_embed = self.doc_encoder(cands)
        # seq_len = cand_embed.shape[2]
        click_embed = click_embed.reshape(num_user, num_click_docs, -1)
        cand_embed = cand_embed.reshape(num_user, num_cand_docs, -1)
        click_embed = click_embed.permute(1, 0, 2)
        click_output, _ = self.mha(click_embed, click_embed, click_embed)
        click_output = click_output.permute(1, 0, 2)
        click_output = F.dropout(click_output, 0.2)
        click_repr = self.proj(click_output)
        click_repr, _ = self.additive_attn(click_repr)
        logits = torch.bmm(click_repr.unsqueeze(1), cand_embed.permute(0, 2, 1)).squeeze(1) # [B, 1, hid], [B, 10, hid]
        if cands.shape[1] > 1:
            return logits
        return torch.sigmoid(logits)
        # return torch.softmax(logits, -1)

    def _get_input_label_from_iter(self, batch_data):
        input_feat = [
            batch_data["clicked_title_token_batch"],
            batch_data["candidate_title_token_batch"],
        ]
        input_label = batch_data["labels"]
        return input_feat, input_label

    def _get_user_feature_from_iter(self, batch_data):
        return batch_data["clicked_title_batch"]

    def _get_news_feature_from_iter(self, batch_data):
        return batch_data["candidate_title_batch"]

    def group_labels(self, labels, preds, group_keys):
        """Devide labels and preds into several group according to values in group keys.

        Args:
            labels (list): ground truth label list.
            preds (list): prediction score list.
            group_keys (list): group key list.

        Returns:
            list, list, list:
            - Keys after group.
            - Labels after group.
            - Preds after group.

        """

        all_keys = list(set(group_keys))
        all_keys.sort()
        group_labels = {k: [] for k in all_keys}
        group_preds = {k: [] for k in all_keys}

        for label, p, k in zip(labels, preds, group_keys):
            group_labels[k].append(label)
            group_preds[k].append(p)

        all_labels = []
        all_preds = []
        for k in all_keys:
            all_labels.append(group_labels[k])
            all_preds.append(group_preds[k])

        return all_keys, all_labels, all_preds

    def fit_model(self, iterator, news, behaviours, optimizer, num_epochs=10):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train()  # set the model in train mode

        for epoch in range(num_epochs):
            train_batches = iterator.load_data_from_file(news, behaviours)
            # initialize progress bar
            progress_bar = tqdm(train_batches, desc=f'Epoch {epoch+1}', ncols=100)
            running_loss = 0.0
            data_size = 0
            total_correct = 0
            total_samples = 0

            for i, batch_data in enumerate(progress_bar):
                # Get the inputs and labels
                inputs, step_labels = self._get_input_label_from_iter(batch_data)
                # inputs = [torch.tensor(inp).to(device) for inp in inputs]
                step_labels = torch.tensor(step_labels).float().to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                step_preds = self.forward(inputs)
                # print('\nPredictions:', step_preds.shape, step_preds)

                # Calculate loss
                loss = self.criterion(step_preds, step_labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                #update data size
                data_size += 1

                running_loss += loss.item()
                step_preds_binary = (step_preds.reshape(-1) > 0.5).int()
                # calculate accuracy
                correct = (step_preds_binary == step_labels.reshape(-1)).sum().item()
                total_correct += correct
                total_samples += step_labels.reshape(-1).size(0)
                # update progress bar
                progress_bar.set_postfix({'Loss': running_loss / data_size, 'Accuracy': total_correct / total_samples})

            # Print statistics
            print('running loss and data size:', running_loss, data_size)
            epoch_loss = running_loss / data_size
            print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')

        print('Finished Training')

    def evaluate_model(self, iterator, news, behaviours):
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      # switch to evaluation mode
      self.eval()
      test_batches = iterator.load_data_from_file(news, behaviours)
      total_loss = 0.0
      total_correct = 0
      total_samples = 0
      data_size = 0

      preds = []
      labels =  []
      imp_indexes = []

      # initialize progress bar
      progress_bar = tqdm(test_batches, desc='Evaluate', ncols=100)

      # we don't need to track gradients for evaluation, so wrap in
      # no_grad to save memory
      with torch.no_grad():
          for batch in progress_bar:
              inputs, step_labels = self._get_input_label_from_iter(batch)
              inputs = [torch.tensor(inp).to(device) for inp in inputs]
              step_labels = torch.tensor(step_labels).float().to(device)

              # forward pass
              step_preds = self.forward(inputs)

              # get impression indexes
              step_imp_idx = batch["impression_index_batch"]

              # calculate loss
              loss = self.criterion(step_preds, step_labels)
              total_loss += loss.item()


              preds.extend(step_preds.reshape(-1))
              labels.extend(step_labels.reshape(-1))
              imp_indexes.extend(np.reshape(step_imp_idx, -1))


              # calculate accuracy
              step_preds_binary = (step_preds.reshape(-1) > 0.5).int()
              correct = (step_preds_binary == step_labels.reshape(-1)).sum().item()
              total_correct += correct
              total_samples += step_labels.size(0)
              data_size += 1

              progress_bar.set_postfix({'Accuracy': total_correct / total_samples, 'Samples Processed': total_samples})

      # return loss and accuracy
      print(f'Loss {total_loss / data_size}, Acc: {total_correct / total_samples:.4f}')

      # move to cpu for further processing
      preds = [p.cpu() for p in preds]
      labels = [l.cpu() for l in labels]


      group_impr_indexes, group_labels, group_preds = self.group_labels(
            labels, preds, imp_indexes)
      return group_impr_indexes, group_labels, group_preds