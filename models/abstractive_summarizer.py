# general imports
import numpy as np
import os
import sys
from collections import Counter
import logging
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from datetime import datetime

# output
import tqdm
import matplotlib.pyplot as plt

# PyTorch imports
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

# imports for preprocessing
import nltk
nltk.download("punkt")
from transformers import BertTokenizer

# other classes/scripts
from evaluation.rouge_evaluator import RougeEvaluator
from models.transformer import Transformer

class AbstractiveSummarizer():
    """
    An Abstractive Summarizer System to summarize articles.
    Based on a Transformer Machine Learning model.
    """
    model = None

    def __init__(self                               
                 , max_article_length: int = 2048               # maximum sequence length for the Encoder input (i.e., input above this threshold will be truncated)
                 , max_summary_length: int = 128                # maximum sequence length for the Decoder input (i.e., input above this threshold will be truncated)
                 , dim_model: int = 256                         # dimensions of the embeddings and hidden layers of the transformer model
                 , nhead= 8                                     # number of attention heads for multi-head attention of the transformer model
                 , num_encoder_layers = 6                       # number of encoder layers of the transformer model
                 , num_decoder_layers = 6                       # number of decoder layers of the transformer model
                 , dropout = 0.1                                # dropout value for transformer model to prevent overfitting, model randomly drops out (= sets to 0) the given amount of features (only during training)
                 , use_bert_vocab: bool = True                  # choose if bert vocab shall be used, if False --> custom vocabulary based on given datasets is built
                 , min_frequency_word_in_vocab: int = 10        # only if custom vocab is used, minimum times a word must appear in the corpus to be added to vocab
                 , gen_method = "top_p"                         # Options: "greedy", "temp_sampling", "beam", "top_k", "top_p"
                 , val_method = "loss"                          # Options: "loss", "rouge" (rouge is computationally significantly more expensive, as it requires the generation of summaries to compare)
                 , temperature = 1                              # "temp_sampling" -> temperature value, # 1 = no change, >1 --> more random output, <1 --> more confident in high probability tokens (increasing likelihood of high prob words)
                 , beam_width = 4                               # "beam" -> number of hypotheses (i.e., possible summaries) to keep at each point
                 , top_k = 10                                   # "top_k" -> number of top k tokens with highest probabilities to keep as basis for new probability distribution that is resampled
                 , top_p = 0.9                                  # "top_p" / "nucleus sampling" -> threshold for the cumulative probabilities subsets of tokens have to reach to be considered
                 , log_level = logging.INFO                     # default log level for logger -> INFO = print to console
                 ):          
        
        # HPO, best result: "dim_model": 128, "nhead": 8, "num_encoder_layers": 2, "num_decoder_layers": 2, "dropout": 0.38603253859327236, "lr": 0.0021570864339274945

        self.model = None

        # set logger
        self.logger = logging.getLogger(__name__)
        self.set_logger(log_level)
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        tensorboard_run_folder = f"runs/{current_time}_dm{dim_model}_nh{nhead}_enc{num_encoder_layers}_dec{num_decoder_layers}_do{int(dropout*100)}"
        self.writer = SummaryWriter(tensorboard_run_folder)

        # set device to train on
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.logger.info(f"Running code on {device}")

        self.path_to_preprocessed_features = None

        #vocab
        self.vocab = None
        self.use_bert_vocab = use_bert_vocab # should bert's vocab be used or should a custom vocab be created based on input?
        if self.use_bert_vocab:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.min_frequency_word_in_vocab = min_frequency_word_in_vocab

        # define transformer variables
        self.max_article_length = max_article_length
        self.max_summary_length = max_summary_length
        self.dim_model = dim_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout

        self.val_method = val_method # Options: "loss", "rouge"
        self.gen_method = gen_method # Options: "greedy", "temp_sampling", "beam", "top_k", "top_p"
        self.temperature = temperature
        self.beam_width = beam_width
        self.top_k = top_k
        self.top_p = top_p

    def set_logger(self, log_level = logging.INFO, to_file = False, log_file = "abstractive_summarizer_log.log") -> None:
        """
        Set up a custom logger to either print output to console or log into log file.
        Can be adjusted for different scenarios such as training and debugging.

        :param log_level: log level to set the logger to (options: logging.INFO, logging.WARNING, logging.DEBUG, logging.ERROR)
        :param to_file: if true, logs are printed to specified file, else to console
        :param log_file: path to log file to print log output to
        """
        # remove old handlers if available
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # set level and handler
        self.logger.setLevel(log_level)
        handler = logging.FileHandler(log_file) if to_file else logging.StreamHandler()
        handler.setFormatter(logging.Formatter("\n%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(handler)

    def load_model(self, model_path: str, vocab_path: str) -> None:
        """
        Load a transformer model based on a given path to model & custom vocab.

        :param model_path: path to model file
        :param vocab_path: path to file with custom vocabulary (not needed in case bert voca is used)
        """

        if not self.use_bert_vocab:
            # load custom vocabulary
            self.vocab = torch.load(vocab_path)
            n_tokens = len(self.vocab)
            pad_id = self.vocab["<PAD>"]
        else: # use bert vocab
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            n_tokens = len(self.tokenizer)
            pad_id = self.tokenizer.pad_token_id

        self.model = Transformer(n_tokens, self.dim_model, pad_id, self.nhead, self.num_encoder_layers, self.num_decoder_layers, self.dropout
                                 , self.max_article_length, self.max_summary_length
                                 , self.logger, self.device).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def train_to_overfit(self, X: list[str], y: list[str], size_to_test = 5, learning_rate=0.001, num_epochs=100) -> None:
        """
        Train a model on a tiny dataset and try to overfit it (i.e., memorize the content & reproduce the summaries).
        Number of epochs should hence be high.
        This method should only be used for debugging purposes, but can provide a variety of relevant insights into the model and the set up pipeline.

        :param X: input (i.e., articles)
        :param y: labels (i.e., summaries)
        :param size_to_test: number of articles/summaries to use to train the model on
        :param learning_rate:
        :param num_epochs:
        """

        X = X[:size_to_test]
        y = y[:size_to_test]
        
        assert len(X) == len(y), "X and y must have the same length"

        # preprocess data & create DataLoader
        dataset = self.preprocess(X, y)
        train_loader = DataLoader(dataset,  shuffle=True, pin_memory=True)

        if not self.use_bert_vocab:
            n_tokens = len(self.vocab)
            pad_id = self.vocab["<PAD>"]
        else:
            n_tokens = len(self.tokenizer)
            pad_id = self.tokenizer.pad_token_id

        self.logger.debug(f" Number of tokens: {n_tokens} | Model dimensions: {self.dim_model}")
        self.model = Transformer(n_tokens, self.dim_model, pad_id, self.nhead, self.num_encoder_layers, self.num_decoder_layers, self.dropout
                                 , self.max_article_length, self.max_summary_length
                                 , self.logger).to(self.device)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for idx, batch in enumerate(train_loader):
                loss = self.compute_loss(batch)
                loss_value = loss.item()
                epoch_loss += loss_value

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.logger.info(f"Epoch {epoch+1}: Loss = {epoch_loss}")

        # try to reproduce summariues
        for i, article in enumerate(X):
            self.logger.info("####################### ORIGINAL SUMMARY #######################")
            self.logger.info( y[i])
            
            self.logger.info("####################### PREDICTED SUMMARY #######################")
            predicted_summary = self.decode(self.generate(article))
            self.logger.info( predicted_summary)

    def train(self, X: list[str], y: list[str], val_X: list[str], val_y: list[str]
              , learning_rate=0.0021570864339274945, batch_size=32, grad_acc=1, num_epochs=20 # 0.001
              , keep_best_n_models=2, lr_scheduler_patience=3, early_stopping_patience=5) -> None:
        """
        Train a Transformer model.

        :param X: list of sentences (i.e., articles)
        :param y: list of sentences (i.e., summaries)
        :param val_X: list of sentences (i.e., articles) to be used in validation
        :param val_y: list of sentences (i.e., summaries) to be used in validation
        :param learning_rate: learning rate for Adam optimizer
        :param batch_size: batch size for training was (32)
        :param grad_acc: number of gradient accumulation steps to sum over. Set to > 1 to simulate larger batch sizes on memory-limited machines.
        :param num_epochs: number of epochs to train for
        :param keep_best_n_models: number of best models to keep (based on validation performance)
        :param lr_scheduler_patience: number of epochs without improvement after which learning rate will be adjusted
        :param early_stopping_patience: numebr of epochs without improvement on the validation dataset after which trainign will stop to prevent overfitting
        """

        assert len(X) == len(y), "X and y must have the same length"

        # load preprocessed data if path is specified, else calculate it
        if self.path_to_preprocessed_features is not None:
            dataset = torch.load(self.path_to_preprocessed_features)
            if not self.use_bert_vocab:
                self.vocab = torch.load("data/vocabulary.pt")
        else:
            dataset = self.preprocess(X, y)

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # setup validation dataset in the same way as training data
        val_dataset = self.preprocess(val_X, val_y) 
        validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        self.logger.debug("Train dataset sample:", dataset[0])
        self.logger.debug("Validation dataset sample:", val_dataset[0])

      # vocabulary related operations
        if not self.use_bert_vocab:
            n_tokens = len(self.vocab)
            pad_id = self.vocab["<PAD>"]
        else:
            n_tokens = len(self.tokenizer)
            pad_id = self.tokenizer.pad_token_id

        self.logger.info(f"Vocabulary size: {n_tokens}")
        self.logger.debug(f" Number of tokens: {n_tokens} | Model dimensions: {self.dim_model}")

        # initialise transformer model and log model graph
        self.model = Transformer(n_tokens, self.dim_model, pad_id, self.nhead, self.num_encoder_layers, self.num_decoder_layers, self.dropout
                                 , self.max_article_length, self.max_summary_length
                                 , self.logger, self.device).to(self.device)
        dummy_input = torch.randint(0, n_tokens, (1, self.max_article_length)).to(self.device) #  create dummy input, so TensorBoard does not have to compute
        self.writer.add_graph(self.model, (dummy_input, dummy_input))                   # log computational model graph
        self.logger.info(f"Embedding layer size: {self.model.embedding.num_embeddings}")
        
        # prepare training start
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        best_model_paths = []
        best_model_scores = []
        if self.val_method == "loss":
            best_validation_score = float("inf")
            scheduler_mode = "min"
        elif self.val_method == "rouge":
            best_validation_score = 0.0
            scheduler_mode = "max"
        scheduler = ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=0.1, patience=lr_scheduler_patience)
        epochs_wo_improvement = 0
        training_losses = []
        validation_losses = []

        # start training loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            with tqdm.tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                for idx, batch in enumerate(train_loader):
                    
                    # Compute the loss:
                    loss = self.compute_loss(batch) / grad_acc
                    loss_value = loss.item()
                    epoch_loss += loss_value

                    # Backprop:
                    loss.backward()

                    # Handle gradient accumulation:
                    if (idx % grad_acc) == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    # write into progressbar
                    pbar.set_postfix({"Batch Loss": loss_value})
                    pbar.update()

                    # collect training loss in writer
                    self.writer.add_scalar("Training Loss", loss_value, epoch * len(train_loader) + idx)

                # calculat eaverage loss for epoch
                avg_epoch_loss = epoch_loss / len(train_loader)
                self.writer.add_scalar("Average Training Loss", avg_epoch_loss, epoch)
                training_losses.append(avg_epoch_loss)

                # evaluate the model:
                validation_score = self.compute_validation_score(val_X, val_y, validation_loader, epoch, self.val_method)
                validation_losses.append(validation_score)
                scheduler.step(validation_score)
                pbar.set_description(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_epoch_loss:.4f} - Val Score: {validation_score:.4f} ")

                with open(f"train_val_loss_{datetime.now().strftime('%Y%m%d_%H%M%S')}", "w") as file:
                    file.write("Epoch,Training Loss,Validation Loss\n")
                    for epoch, (train_loss, val_loss) in enumerate(zip(training_losses, validation_losses), 1):
                        file.write(f"{epoch},{train_loss},{val_loss}\n")

                # see if model has improved based on selected validation method
                if self.val_method == "loss":
                    improved = validation_score < best_validation_score
                    score_condition = len(best_model_scores) < keep_best_n_models or validation_score < min(best_model_scores)
                elif self.val_method == "rouge":
                    improved = validation_score > best_validation_score
                    score_condition = len(best_model_scores) < keep_best_n_models or validation_score > max(best_model_scores)

                # early stopping check & model save
                if improved:
                    best_validation_score = validation_score
                    epochs_wo_improvement = 0

                    # Save the model, if performance has improved (keeping n models saved)
                    if score_condition:
                        # Save the model:
                        best_model_scores.append(validation_score)
                        best_model_paths.append(f"model-{str(epoch)}_{self.val_method}-score-{str(validation_score)}-{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
                        torch.save(self.model.state_dict(), best_model_paths[-1])

                        # Delete the worst model:
                        if len(best_model_scores) > keep_best_n_models:
                            worst_model_index = np.argmax(best_model_scores)
                            
                            os.remove(best_model_paths[worst_model_index])
                            del best_model_paths[worst_model_index]
                            del best_model_scores[worst_model_index]
                else:
                    epochs_wo_improvement += 1
                    if epochs_wo_improvement >= early_stopping_patience:
                        # in case of rouge: only break in case rouge is not 0 (need some time to learn good representation)
                        if not (self.val_method == "rouge" and best_validation_score == 0.0):
                            break

        # plot loss behaviour
        num_epochs_plotted = min(len(training_losses), len(validation_losses))

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs_plotted + 1), training_losses[:num_epochs_plotted], label="Training Loss", color="steelblue", marker="o")
        plt.plot(range(1, num_epochs_plotted + 1), validation_losses[:num_epochs_plotted], label="Validation Loss", color="darkorange", marker="o")
        plt.title("Training and Validation Losses", fontsize=14)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        plt.savefig("training_validation_loss.pdf", format="pdf")

        # Save the plot
        plt.savefig("training_validation_loss.pdf", format="pdf")

        # Recall the best model:
        best_model_index = np.argmax(best_model_scores)
        self.model.load_state_dict(torch.load(best_model_paths[best_model_index]))

        self.writer.close()

    def preprocess(self, X: list[str], y: list[str]) -> TensorDataset:
        """
        Preprocessing method to tokenize articles and summaries, add special tokens & right-shift data for teacher forcing.

        :param X: list sentences (i.e., articles)
        :param y: list of sentences (i.e., summaries)
        :return: TensorDataset containing the preprocessed X & corresponding y values
        """

        # STEP I - TOKENIZATION
        tokenized_articles = []
        tokenized_summaries = []

        for article, summary in tqdm.tqdm(zip(X, y), desc="Tokenize articles & summaries", total=len(X)):

            # ARTICLE
            # tokenize & introducen <EOS> token
            article_tokens = self.tokenize_text(article)
            if self.use_bert_vocab:
                article_tokens.append(self.tokenizer.sep_token) # use berts [SEP] as EOS token
            else:
                article_tokens.append("<EOS>")

            # SUMMARY
            summary_tokens = self.tokenize_text(summary)
            if self.use_bert_vocab:
                summary_tokens.append(self.tokenizer.sep_token) # use berts [SEP] as EOS token
            else:
                summary_tokens.append("<EOS>")
           
            tokenized_articles.append(article_tokens)
            tokenized_summaries.append(summary_tokens)
        
        self.logger.debug("Sample tokenized article:", tokenized_articles[0])
        self.logger.debug("Sample tokenized summary:", tokenized_summaries[0])

        if not self.use_bert_vocab:
            # STEP II - CREATE VOCABULARY
            if self.vocab is None:
                # create vocabulary if non existing
                self.create_vocab((tokenized_articles + tokenized_summaries)) # get vocab dict
            else:
                # update vocabulary if it already exists
                self.update_vocab((tokenized_articles + tokenized_summaries))

            # allow maximum, if no max_length defined
            if self.max_article_length is None:
                self.max_article_length = max(len(article) for article in tokenized_articles)
            if self.max_summary_length is None:
                self.max_summary_length = max(len(summary) for summary in tokenized_summaries)

        # STEP III - PAD TEXTS TO MAX_LENGTH
        padded_articles = [self.pad_and_convert_to_ids(text, self.max_article_length) for text in tokenized_articles]

        # STEP IV - TEACHER FORCING -> RIGHT-SHIFT SUMMARIES
        start_token_id = self.tokenizer.sep_token_id if self.use_bert_vocab else self.vocab["<EOS>"] # use <EOS> token in the beginning
        padded_shifted_summaries = []
        for summary in tokenized_summaries:
            shifted_summary = [start_token_id] + self.pad_and_convert_to_ids(summary, self.max_summary_length)[:-1]  # Add EOS token and remove the last token, effectively right-shifting the data
            padded_shifted_summaries.append(shifted_summary)

        self.logger.debug("Padded articles shape:", len(padded_articles))
        self.logger.debug("Padded summaries shape:", len(padded_shifted_summaries))

        # create dataset consisting of preprocessed X (articles) & y (summaries)
        dataset = TensorDataset(torch.tensor(padded_articles), torch.tensor(padded_shifted_summaries))

        # save dataset & vocabulary
        torch.save(dataset, "data/preprocessed_dataset.pt")
        torch.save(self.vocab, "data/vocabulary.pt")

        return dataset


    def tokenize_text(self, text_to_tokenize: str) -> list[str]:
        """
        Simple method to tokenize text by eitehr using nltk or bert tokenizer.

        :param text_to_tokenize: text that shall be tokenized and returned
        :return: list of tokens
        """
        if self.use_bert_vocab:
            return self.tokenizer.tokenize(text_to_tokenize)

        return nltk.word_tokenize(text_to_tokenize)
    
        # Also tested, but discared due to worse performance for task at hand:
        sentences = nltk.sent_tokenize(text_to_tokenize)
        tokenized_text = []
        # add <EOS> tag to each sentence 
        for sentence in sentences:
            # tokens = ["<SOS>"]
            # tokens.extend(nltk.word_tokenize(sentence))
            tokens = nltk.word_tokenize(sentence)
            # tokens.append("<EOS>")
            tokenized_text.extend(tokens)

        return tokenized_text
    
    def create_vocab(self, tokenized_texts: list[list[str]]) -> None:
        """
        Create a vocabulary based on a list of tokenized texts (e.g., articles & summaries).

        :param tokenized_texts: list of tokenized texts, each containing tokens
        """
        # count token frequencies
        token_counts = Counter(token for text in tokenized_texts for token in text)
        
        # filter tokens based on their frequency in the corpus (i.e., add token and count for every token whose count exceeds threshold)
        frequent_tokens = {token for token, count in token_counts.items() if count >= self.min_frequency_word_in_vocab}
        # frequent_tokens.update(["<UKN>", "<PAD>", "<SOS>", "<EOS>", "<End of Summary>"])  # include special tokens (see above)
        frequent_tokens.update(["<UKN>", "<PAD>", "<EOS>"])  # include special tokens in vocab

        # create vocabulary mapping token to index (just enumerated)
        self.vocab = {token: index for index, token in enumerate(frequent_tokens)}
        self.logger.debug(self.vocab)

    def update_vocab(self, tokenized_texts: list[list[str]]) -> None:
        """
        Update an already existing custom vocabulary in case one already exists, but need to be amended.

        :param tokenized_texts: list of tokenized texts, each containing tokens
        """
        # count token frequencies in new texts
        new_token_counts = Counter(token for text in tokenized_texts for token in text)

        # filter new tokens based on their frequency in the corpus (i.e., add token and count for every token whose count exceeds threshold) and whether they are already in vocab
        unique_new_tokens = {token for token, count in new_token_counts.items() if count >= self.min_frequency_word_in_vocab and token not in self.vocab}

        # update vocabulary with new tokens
        current_max_index = max(self.vocab.values())
        new_index = current_max_index + 1

        for token in unique_new_tokens:
            self.vocab[token] = new_index
            new_index += 1

    def pad_and_convert_to_ids(self, tokens: list[str], max_length: int) -> list[int]:
        """
        Pad a given list of tokens to max lenght and convert the tokens to ids basde on the used vocabulary.

        :param tokens: list of tokens to pad and convert
        :param max_length: maximum length to truncate to or pad to
        :return: list of of token ids
        """
        if self.use_bert_vocab:
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens) # convert all tokens into ids using berts inbuilt function
            padded_token_ids = token_ids[:max_length] + [self.tokenizer.pad_token_id] * max(max_length - len(token_ids), 0) # use given ids and add padding if not reached max lenght yet
        else:
            # pad, if necessary, trim to max_length
            padded_text = tokens[:max_length] + ["<PAD>"] * max(max_length - len(tokens), 0)  #  add padding if not reached max lenght yet
            # convert to ids using dicts
            padded_token_ids = [self.vocab.get(token, self.vocab["<UKN>"]) for token in padded_text] # convert to ids using custom vocab
        return padded_token_ids

    def compute_loss(self, batch) -> torch.Tensor:
        """
        Compute cross-entropy loss based on batch.

        :param batch: tensor of token indices. Dimensionality is (batch_size, sequence_length)
        :return: Cross Entropy loss
        """

        # split batch into input and output tensors
        input_ids, shifted_output_ids = batch
        input_ids, shifted_output_ids = input_ids.to(self.device), shifted_output_ids.to(self.device)

        # make predictions using transformer model
        predicted_outputs = self.model(input_ids, shifted_output_ids)

        # permute predicted_outputs to match mask dimensions: [batch_size, sequence_length, n_tokens]
        predicted_outputs = predicted_outputs.permute(1, 0, 2)  # new shape: [batch_size, sequence_length, n_tokens]

        # create mask for non-PAD tokens in shifted output --> necessary to ensure pad tokens are not considered for loss calculation!
        pad_id = self.tokenizer.pad_token_id if self.use_bert_vocab else self.vocab["<PAD>"]
        mask = (shifted_output_ids[:, 1:] != pad_id)  # shape of mask: [batch_size, sequence_length - 1], shifted by one token because of teacher forcing

        # expand mask to match dimensions of predicted_outputs (see above)
        expanded_mask = mask.unsqueeze(-1)  # shaoe of mask: [batch_size, sequence_length - 1, 1], adding last dimension
        expanded_mask = expanded_mask.expand_as(predicted_outputs[:, :-1, :])  # align with predicted_outputs dimensions, so it can be applied element-wise -> last predicted token does not have next token in targets (remove it)!
        
        # apply (expanded) mask to predicted outputs (i.e, without padding) by 1. selecting all tokens except the last one, 2. applying the mask, 3. reshaping the filtered predictions into 2d tensor
        valid_predicted_outputs = predicted_outputs[:, :-1, :][expanded_mask]
        valid_predicted_outputs = valid_predicted_outputs.view(-1, predicted_outputs.size(-1))  # reshape to [num_valid_tokens, n_tokens]
        # Prepare target ids
        output_ids = shifted_output_ids[:, 1:].contiguous().view(-1)  # select reference/traget tokens (instead of first one because of teacher forcing) and reshape to 1d tensor
        output_ids = output_ids[mask.view(-1)] # apply original mask (see above= to filter out all pad tokens)

        # return loss
        return F.cross_entropy(valid_predicted_outputs, output_ids)
    
    def compute_validation_score(self, X: list[str], y: list[str], val_loader, current_epoch: int, val_method: str) -> float:
        """
        Compute a validation score (ROUGE-based or corss entropy loss based).

        :param X: list of sentences (i.e., articles)
        :param y: list of sentences (i.e., summaries)
        :param val_loader: DataLoader with validation data
        :param current_epoch: current epoch (used for writer)
        :param val_method: vlidation method to use ("loss" for Cross Entropy Loss based evluatio or "rouge" for ROUGE score based evaluation)

        :return: loss value
        """

        self.model.eval()
        with torch.no_grad():
            # based on compute loss function
            if val_method == "loss":
                total_loss = 0
                for batch in val_loader:
                    loss = self.compute_loss(batch)
                    total_loss += loss.item() 

                # calc and return average loss
                average_loss = total_loss / len(val_loader)
                self.writer.add_scalar("Validation Loss", average_loss, current_epoch)
                return average_loss
            # based on ROUGE score
            elif val_method == "rouge":
                predicted_summaries = []

                for article in X:
                    predicted_summary = self.decode(self.generate(article))
                    predicted_summaries.append(predicted_summary)

                evaluator = RougeEvaluator()
                scores = evaluator.batch_score(predicted_summaries, y)

                # calc average rouge score
                average_rouge_score = sum([score["f"] for score in scores.values()]) / len(scores)
                self.writer.add_scalar("Validation ROUGE", average_rouge_score, current_epoch)
                return average_rouge_score

    def generate(self, article: str, min_summary_length = 10, max_summary_length=100) -> list[int]:
        """
        Given an article, generate a summary.
        Method forwards calculations to Generator class.

        :param article: article that shall be summarized
        :param min_summary_length: minimum length of the summary to be generated
        :param max_summary_length: maximum length of the summary to be generated

        :return: list of token indices representing the generated summary
        """

        self.model.eval()
        with torch.no_grad():
            tokenized_article = self.tokenize_text(article)
            input_ids = self.pad_and_convert_to_ids(tokenized_article, self.max_article_length)
            input_tensor = torch.tensor([input_ids]).to(self.device)
            eos_id = self.tokenizer.sep_token_id if self.use_bert_vocab else self.vocab["<EOS>"]
            
            return Generator(model=self.model, gen_method=self.gen_method, input_tensor=input_tensor, device=self.device, eos_id=eos_id
                             , min_summary_length=min_summary_length, max_summary_length=max_summary_length
                             , temperature=self.temperature, beam_width=self.beam_width, top_k=self.top_k, top_p= self.top_p
                             ).gen_summary()

    def decode(self, tokens) -> str:
        """
        Decode tokens into a string.

        :param tokens: list of token indices or single token

        :return: decoded tokens to string
        """
        if isinstance(tokens, int):
            tokens = [tokens]

        if self.use_bert_vocab:
            # use BERT's tokenizer's decode method
            decoded_string = self.tokenizer.decode(tokens)
        else:
            # invert vocabulary (necessary to decode back to token)
            inverted_vocab = {index: token for token, index in self.vocab.items()}
            decoded_tokens = [inverted_vocab.get(index, "<UKN>") for index in tokens] # put ukn token for every token not found in vocab

            # form decoded string by joinin tokens
            decoded_string = ' '.join(decoded_tokens)

        return decoded_string

    def predict(self, X: list[str], k=3):
        """
        Given a list of articles, predict summaries.

        X: list of articles
        """   
        for article in tqdm.tqdm(X, desc="Running abstractive summarizer"):
            output_token_indices = self.generate(article)
            summary = self.decode(output_token_indices)

            self.logger.debug("################### ARTICLE ###################\n")
            self.logger.debug(article)

            self.logger.debug("################### SUMMARY ###################\n")
            self.logger.debug(summary)
            
            yield summary

    def objective(self, trial: optuna.Trial, train_loader, validation_loader, val_X, val_y, n_tokens, pad_id, grad_acc = 1, num_epochs=20, lr_scheduler_patience=3, early_stopping_patience=5) -> float:
        """
        Perform hyperparameter optimisation trial and return validation score.
        Effectively a reduced and adjusted train function
        https://optuna.readthedocs.io/en/stable/index.html
        

        :param trial: optuna trial object for hyperparameter optimisation
        :param train_loader: DataLoader for training data
        :param validation_loader: DataLoader for validation data
        :param val_X: validation input
        :param val_y: validatiom label
        :param n_tokens: number of tokens in vocab
        :param pad_id: token id of padding token
        :param grad_acc: gradient acc steps
        :param num_epochs: number of training epochs to run
        :param lr_scheduler_patience: patience for learning rate scheduler
        :param early_stopping_patience: patience for early stopping

        :return: validation score of model for current trial run
        """
         
        # hyperparameters to be optimised
        dim_model = trial.suggest_categorical("dim_model", [128, 256, 512])
        nhead = trial.suggest_categorical("nhead", [4, 8])
        num_encoder_layers = trial.suggest_int("num_encoder_layers", 1, 4)
        num_decoder_layers = trial.suggest_int("num_decoder_layers", 1, 4)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        learning_rate = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        
        self.model = Transformer(n_tokens, dim_model, pad_id, nhead, num_encoder_layers, num_decoder_layers, dropout
                                 , self.max_article_length, self.max_summary_length
                                 , self.logger, self.device).to(self.device)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        best_validation_score = float("inf")
                
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=lr_scheduler_patience)
            
        epochs_wo_improvement = 0
    
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            # with tqdm.tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for idx, batch in enumerate(train_loader):
                
                # Compute the loss:
                loss = self.compute_loss(batch) / grad_acc
                loss_value = loss.item()
                epoch_loss += loss_value

                # Backprop:
                loss.backward()

                # Handle gradient accumulation:
                if (idx % grad_acc) == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                # pbar.set_postfix({"Batch Loss": loss_value})
                # pbar.update()

            # avg_epoch_loss = epoch_loss / len(train_loader)
    
            # Evaluate the model:
            validation_score = self.compute_validation_score(val_X, val_y, validation_loader, epoch, self.val_method)
            scheduler.step(validation_score)
            # pbar.set_description(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_epoch_loss:.4f} - Val Score: {validation_score:.4f} ")

            # report score for pruning
            trial.report(validation_score, epoch)

            # handle pruning 
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # early stopping check & model save
            if validation_score < best_validation_score:
                best_validation_score = validation_score
                epochs_wo_improvement = 0
            else:
                epochs_wo_improvement += 1
                if epochs_wo_improvement >= early_stopping_patience:
                    break
        
        return validation_score

    def write_trial_result_to_file(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial, filename="hpo_trial_results.txt"):
        """
        Callback function to write result of each trial to a file.

        :param study: Optuna study
        :param trial: Optuna trial
        :param filename: Name of the file to save the trial results.
        """
        with open(filename, "a") as file:
            file.write(f"Trial {trial.number}: Score={trial.value}, Params={trial.params}\n")

                
    def hyperparameter_search(self, X: list[str], y: list[str], val_X: list[str], val_y: list[str], n_trials=20, batch_size=32) -> None:
        """
        Initialise a hyperparameter search based on optuna (see https://optuna.readthedocs.io/en/stable/index.html).

        :param X: training data input
        :param y: training data labels
        :param val_X: validation data input
        :param val_y: validation data labels
        :param n_trials: number of test runs ("trials")
        :param batch_size: batch size 
        """
        setting_val_method = self.val_method

        if setting_val_method != "loss":
            setting_val_method = "loss"

        # load preprocessed data if path is specified, else calculate it
        if self.path_to_preprocessed_features is not None:
            dataset = torch.load(self.path_to_preprocessed_features)
            if not self.use_bert_vocab:
                self.vocab = torch.load("data/vocabulary.pt")
        else:
            dataset = self.preprocess(X, y)

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # setup validation dataset in the same way as training data
        val_dataset = self.preprocess(val_X, val_y) 
        validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        self.logger.debug("Train dataset sample:", dataset[0])
        self.logger.debug("Validation dataset sample:", val_dataset[0])

      
        if not self.use_bert_vocab:
            n_tokens = len(self.vocab)
            pad_id = self.vocab["<PAD>"]
        else:
            n_tokens = len(self.tokenizer)
            pad_id = self.tokenizer.pad_token_id

        self.logger.info(f"Vocabulary size: {n_tokens}")

        self.logger.debug(f" Number of tokens: {n_tokens} | Model dimensions: {self.dim_model}")

        # specify pruner and sampler
        sampler = TPESampler()
        pruner = MedianPruner()

        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)


        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self.objective(trial, train_loader, validation_loader, val_X, val_y, n_tokens, pad_id)
                       , n_trials=n_trials
                       , callbacks= [lambda study, trial: self.write_trial_result_to_file(study, trial)]
                       , show_progress_bar=True)
        self.logger.info("Best hyperparameters: ", study.best_params)

        # reset variable to value before
        self.val_method = setting_val_method

class Generator():
    """
    Custom generator class encapsulating different generation/decoding startegies for the abstractive summarizer model.
    Generation methods can be chosen from the following:
    - "greedy": choosing the token with the highest probability
    - "beam": use a beam search to collect multiple different hypotheses for summaries and choose the best one in the end
    - "temp_sampling": sampling from the probability distribution (i.e., randomly picking a token)
                        , a temperature value can be applied to increase (temperature < 1) or decrease (temperature > 1) the likelihood of high probability words
                        , default: 1 --> no adjustment of the distribution
    - "top_k": only sample from the top k tokens with highest probabilities
    - "top_p": sample from the smallest subset of tokens whose cumulative probabilitie is higher than threshold value p (also known as nuclues sampling)
    """
    def __init__(self, model, gen_method, input_tensor, device, eos_id, min_summary_length = 10, max_summary_length = 100
                 , temperature = 1, beam_width = 4, top_k=10, top_p=0.9):
        self.model = model
        self.gen_method = gen_method # greedy, temp_sampling, beam, top_k, top_p
        self.input_tensor = input_tensor
        self.device = device
        self.eos_id = eos_id
        self.min_summary_length = min_summary_length
        self.max_summary_length = max_summary_length
        self.temperature = temperature 
        self.beam_width = beam_width
        self.top_k = top_k
        self.top_p = top_p
    
    def gen_summary(self) -> list[int]:
        """
        Generate a summary for the input tensor.

        :return: list of token ids, representing summary
        """

        if self.gen_method == "beam":
            return self.beam_search_generate()
        elif self.gen_method in ["greedy", "temp_sampling", "top_k", "top_p"]:
            return self.generate()

        else:
            raise Exception("Please choose a valid generation method.")

    
    def generate(self) -> list[int]:
        """
        Generate method to handle "greedy", "temp_sampling", "top_k", "top_p" generation methods.

        :return: list of token ids, representing summary
        """

        # start with predicting, start with manually introduced start token
        output_ids = [self.eos_id]
        output_tensor = torch.tensor([output_ids], device=self.device)

        while len(output_ids) < self.max_summary_length:

            model_output = self.model(self.input_tensor, output_tensor)
            model_predictions = model_output[-1, 0, :]

            if len(output_ids) < self.min_summary_length:
                model_predictions[self.eos_id] = -float("Inf") # don't predict eos before min len is reached!

            match self.gen_method:
                case "temp_sampling":
                    if self.temperature != 0.0: # if temperature is defined, adjust greediness of model by dividing logits through temperature
                        model_predictions = model_predictions / self.temperature
                    # get probability distribution
                    probabilities = F.softmax(model_predictions, dim=-1)
                    # sample from distribution
                    next_token = torch.multinomial(probabilities, 1).item()
                case "top_k":
                    top_k_values, top_k_indices = torch.topk(model_predictions, k=self.top_k) # pick top k values from predictions
                    probabilities = F.softmax(top_k_values, dim=-1)
                    next_token = top_k_indices[torch.multinomial(probabilities, 1)].item() # sample based on these values
                case "top_p": 
                    desc_sorted_predictions, desc_sorted_indices = torch.sort(model_predictions, descending=True)
                    probabilities = F.softmax(desc_sorted_predictions, dim=-1)
                    cumulative_probabilities = torch.cumsum(probabilities, dim=-1)

                    # keep smallest subset of tokens whose cum prob is at least p
                    filtered_indices = desc_sorted_indices[cumulative_probabilities <= self.top_p]
                    filtered_probabilities = probabilities[cumulative_probabilities <= self.top_p]

                    filtered_probabilities = filtered_probabilities / torch.sum(filtered_probabilities) # renormalize to ensure sum of all probs = 1

                    if len(filtered_indices) > 0:
                        next_token = filtered_indices[torch.multinomial(filtered_probabilities, 1)].item()
                    else: # in case all probs are filtered out, just fallback to most probable token as next token
                        next_token = desc_sorted_indices[0].item()

                case _: # greedy as default!
                    next_token = model_predictions.argmax(dim=-1).item()  # choose next token based on highest probability

            # stop if eos is reached
            if next_token == self.eos_id:
                break

            # else append next token
            output_ids.append(next_token)
            output_tensor = torch.tensor([output_ids], device=self.device)

        return output_ids[1:]  # Exclude the initial start token

    
    def beam_search_generate(self) -> list[int]:
        """
        Generate a summary based on a beam search approach.

        :return: list of token ids, representing summary
        """
    
        # beam = list of tuples (score, token id sequence), start with eos token
        beams = [(0, [self.eos_id])]

        complete_summaries = []

        # run until max summary length is reached
        for i in range(self.max_summary_length):

            # go through each beam and expand with new token candidates
            new_beams = []
            for score, token_ids in beams:

                # create output tensor (starting with eos token initialized above) & get model predictions
                output_tensor = torch.tensor([token_ids], device=self.device)
                output = self.model(self.input_tensor, output_tensor)

                # select output and convert into probabilities, before selecting the correspondong beam width top elements as candidates
                model_predictions = output[-1, 0, :] # select last token in output sequence, focus on one/first sequence only, get all predictions
                if len(token_ids) < self.min_summary_length:
                    model_predictions[self.eos_id] = -float("Inf") # don't predict eos before min len is reached!
                
                log_probs = F.log_softmax(model_predictions, dim=-1) #softmax on last dimension (logits)
                next_tokens = log_probs.topk(self.beam_width)[1] # get top beam_width token indices

                # update list of beams (i.e., candidate sequences) --> add possible tokens to each current beam
                for new_possible_score, new_possible_token in zip(log_probs, next_tokens):
                    # get scalars from tensors
                    new_possible_score = new_possible_score.item()
                    new_possible_token = new_possible_token.item()

                    # check for eos token
                    if new_possible_token == self.eos_id:
                        complete_summaries.append((score + new_possible_score, token_ids)) # don't add eos token to final summary
                    else:
                        new_beam = (score + new_possible_score, token_ids + [new_possible_token]) # add eos token --> new beam = new score + new sequence
                        new_beams.append(new_beam)

            # sort new beams by scores (tuple[0]) and take top beam_widt beams to proceed with
            sorted_beams = sorted(new_beams, key=lambda beam_tuple: beam_tuple[0], reverse=True)
            beams = sorted_beams[:self.beam_width]

            if not beams: # stop if no active beams
                break

        # if available, take the best summary and return it without its eos start token
        if complete_summaries:
            best_summary_tuple = max(complete_summaries, key=lambda beam_tuple: beam_tuple[0]) # best summary by beam score
            best_summary = best_summary_tuple[1] # get token ids
        else:
            best_summary = beams[0][1] #if no complete summaries, take first beam's token ids as summary
        return best_summary[1:] 