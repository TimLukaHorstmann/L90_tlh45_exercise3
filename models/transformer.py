import math
import logging
import torch
import torch.nn as nn       

class Transformer(nn.Module):

    def __init__(self, n_tokens, dim_model, pad_token_id
                 , nhead=4, num_encoder_layers = 2, num_decoder_layers = 2, dropout = 0.1
                 , max_article_length=2048, max_summary_length=128
                 , logger: logging.Logger =None
                 , device=None):
        
        super().__init__()

        self.logger=logger
        self.device = device

        # add context to model
        self.model_type = "Transformer"
        self.pad_token_id = pad_token_id

        # set up transformer components
        # 1. Word Embeddings
        self.dim_model = dim_model
        self.embedding = nn.Embedding(n_tokens, dim_model)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(dim_model=dim_model, dropout=dropout, max_article_length=max_article_length, max_summary_length=max_summary_length, logger=self.logger, device=self.device)
        
        # 3. Transformer (Encoder-Decoder Structure)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            dim_feedforward=1024
        )

        # 4. Fully connected output layer
        self.output = nn.Linear(dim_model, n_tokens)
         
    def forward(self, input, output):
        """
        Forward pass through transformer model.

        :param input: input token ids tensor (batch_size, input sequence length)
        :param output: output token ids tensor (batch_size, output sequence length)
        :return: output tensor from the transformer
        """

        # Embedding & positional encoding (batch_size, sequence length, n_features)
        input_embedded = self.embedding(input) * math.sqrt(self.dim_model) # get word embeddings for input, apply sqrt to prevent dot product from getting too large
        output_embedded = self.embedding(output) * math.sqrt(self.dim_model)

        self.logger.debug("Input embedding shape:", input_embedded.shape)
        self.logger.debug("Output embedding shape:", output_embedded.shape)

        input_encoded = self.pos_encoder(input_embedded, sequence_type="article")
        output_encoded = self.pos_encoder(output_embedded, sequence_type="summary")

        # Change shape to match PyTorch's transformer module (sequence length, batch_size, dim_model),
        input_encoded = input_encoded.permute(1, 0, 2)
        output_encoded = output_encoded.permute(1, 0, 2)

        # Create masks to ignore <PAD> tokens by setting them to extremely low value (-inf)
        input_padding_mask = (input == self.pad_token_id)
        target_padding_mask = (output == self.pad_token_id)
        # Generate mask to ignore subsequent tokens --> enfore auto-regressive behaviour of model (i.e., preds only depend on previous tokens)
        target_mask = self.generate_subsequent_mask(output_encoded.size(0)).to(self.device)


        self.logger.debug("input_pad_mask shape:", input_padding_mask.shape, " Type:", input_padding_mask.dtype)
        self.logger.debug("target_pad_mask shape:", target_padding_mask.shape, " Type:", target_padding_mask.dtype)
        self.logger.debug("target_mask shape:", target_mask.shape, " Type:", target_mask.dtype)

        # Apply masks (for padding & for subsequent tokens) to transformer model output
        transformer_output = self.transformer(
            input_encoded,
            output_encoded,
            src_key_padding_mask=input_padding_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=input_padding_mask,
            tgt_mask=target_mask
        )


        output = self.output(transformer_output)
        return output

    def generate_subsequent_mask(self, size: int) -> torch.Tensor:
        """
        Generate square mask to square out future positions. (--> do not allow look ahead)
        Masked positions: filled with -inf
        Unmasked positions: filled with 0.0)

        :param size: size of the square mask
        :return: mask
        """
        # upper tringual matrix (filled with 1s)
        upper_tri_matrix = torch.triu(torch.ones(size, size))

        # transpose to align & convert to floats
        upper_tri_matrix = upper_tri_matrix.transpose(0, 1)
        upper_tri_matrix = upper_tri_matrix.float()

        # refill lower tri with -inf instead of default 0s
        mask_with_inf = upper_tri_matrix.masked_fill(upper_tri_matrix == 0, float("-inf"))

        # refill upper tri with 0s instead of 1 --> allow attention to these positions
        mask = mask_with_inf.masked_fill(mask_with_inf == 1, float(0.0))

        return mask
    
class PositionalEncoding(nn.Module):
    """
    Positional Encoding component.
    """
    def __init__(self, dim_model, dropout, max_article_length = 2048, max_summary_length = 128, logger: logging.Logger =None, device = None):
        super().__init__()

        self.logger = logger
        self.device = device

        # define model state by setting dropout as regularization technique
        self.dropout = nn.Dropout(dropout)

        # hanlde length for article and summary differently (because of significant lenght differences)
        self.pos_encoding_article = self.create_encoding(dim_model, max_article_length)
        self.pos_encoding_summary = self.create_encoding(dim_model, max_summary_length)
    
    def create_encoding(self, dim_model, max_length) -> torch.Tensor:
        """
        Calculate positional encoding differently for specified max_length.

        :param dim_model: the model dimensions (must match embedding dimensions)
        :param max_length: the max length to use for the positional encoding (positions)

        :return: positional enoding
        """
        # encoding 
        pos_encoding = torch.zeros(max_length, dim_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)

        # scaling factor to be used in encoding --> following "Attention is all you need paper" -> 1000^(2i/d_model)
        scaling_factor = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) 
        
        # PE(pos, 2i) = sin(pos/1000^(2i/d_model))
        pos_encoding[:, 0::2] = torch.sin(position * scaling_factor) #even indices --> sine
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/d_model))
        pos_encoding[:, 1::2] = torch.cos(position * scaling_factor) # odd indices --> cosine
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1) # add extra dimension -> 3D tensor, and swap first two dimensions -> fit batch dimensions of PyTorch
        pos_encoding.to(self.device)
        self.register_buffer(f"pos_encoding_{max_length}",pos_encoding) # register as buffer to not get updated during training

        return pos_encoding


    def forward(self, embeddings: torch.tensor, sequence_type: str) -> torch.tensor:
        """
        Apply positional encoding to given embeddings.

        :param embeddings: input embeddings tensor
        :paam sequence_type: specify if input is article or summary
        :return: embedding tensor with added positional encodings
        """

        self.logger.debug("Input embeddings shape:", embeddings.shape)

        # residual connection + pos encoding (dependet on lengths, article vs. summary)
        if sequence_type == "article":
            pos_encoding = self.pos_encoding_article[:embeddings.size(0), :] # make sure sequences are aligned & apply positional encoding to every present embedding token
        else: #summary
            pos_encoding = self.pos_encoding_summary[:embeddings.size(0), :] # make sure sequences are aligned & apply positional encoding to every present embedding token

        # add positional encoding to embeddings
        combined_encoding = embeddings + pos_encoding.to(self.device)
        
        self.logger.debug("Combined encoding shape:", combined_encoding.shape)

        return self.dropout(combined_encoding) # apply dropout factor for regularization