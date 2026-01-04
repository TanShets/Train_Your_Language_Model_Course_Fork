import sys
sys.path.append('..')

from minbpe.gpt4 import GPT4Tokenizer
from minbpe.basic import BasicTokenizer
import torch
from torch import nn
from torch.nn import functional as F
import math
from typing import Optional

tokenizer = BasicTokenizer()
tokenizer.load(model_file='../output/tokenizer/temp_tokenizer.model')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
context_size = 512
block_size = 512
embedding_dimension = 256
no_of_attention_heads = 8
key_query_reduced_dimensionality = 8
no_of_blocks = 16
vocab_size = len(tokenizer.vocab)+len(tokenizer.special_tokens)

class AttentionHead(nn.Module):
    def __init__(self, key_query_reduced_dimensionality, embedding_dimension):
        super().__init__()
        self.key_query_reduced_dimensionality = key_query_reduced_dimensionality
        self.embedding_dimension = embedding_dimension
        self.query = nn.Linear(self.embedding_dimension, self.key_query_reduced_dimensionality, bias = False)
        self.key = nn.Linear(self.embedding_dimension, self.key_query_reduced_dimensionality, bias = False)
        self.value = nn.Linear(self.embedding_dimension, self.key_query_reduced_dimensionality, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, embeddedVector: torch.Tensor) -> torch.Tensor:
        _, T, _ = embeddedVector.shape
        Q = self.query(embeddedVector)
        K = self.key(embeddedVector)
        V = self.value(embeddedVector)

        KT = torch.transpose(K, 1, 2)
        R = torch.matmul(KT, Q) / (self.embedding_dimension**0.5)
        R = R.masked_fill(torch.transpose(torch.tril(R), 1, 2) == 0, float('-inf'))
        R = F.softmax(R, dim = -1)
        change_in_embedded_vector = V @ R

        return change_in_embedded_vector

class AttentionLayer(nn.Module):
    def __init__(self, no_of_attention_heads, key_query_reduced_dimensionality, embedding_dimension):
        super().__init__()
        self.no_of_attention_heads = no_of_attention_heads
        self.key_query_reduced_dimensionality = key_query_reduced_dimensionality
        self.embedding_dimension = embedding_dimension
        self.heads = nn.ModuleList([AttentionHead(key_query_reduced_dimensionality, embedding_dimension) for _ in range(no_of_attention_heads)])
        # Originally We could have considered each Value Weighr Wv to be (embedded_dimension, embedded_dimension) but that's too massive so we can alt do (embedded_dimension, reduced_dim) x (reduced_dim, embedded_dimension)
        # That's what we do in actual transformers, so that's (Output Weight) x (Value Weight new), Value Weight goes to each head and Output Weight is just joined together for all heads
        self.output_weight = nn.Linear(self.no_of_attention_heads * self.key_query_reduced_dimensionality, self.embedding_dimension)
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, embeddedVector: torch.Tensor) -> torch.Tensor:
        output = torch.cat([attentionHead(embeddedVector) for attentionHead in self.heads], dim = -1)
        output = self.dropout(self.output_weight(output))
        return output

#Multilayer Perceptron Layer
class MLPLayer(nn.Module):
    def __init__(self, embedding_dimension, expansion_factor = 4):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.expansion_factor = expansion_factor

        #Expanding Layer which by default will be set to 4 times the size of the embedded dimension
        self.neural_net = nn.Sequential(
            nn.Linear(self.embedding_dimension, self.expansion_factor * self.embedding_dimension),
            nn.ReLU(),
            nn.Linear(self.expansion_factor * self.embedding_dimension, self.embedding_dimension),
            nn.Dropout(p = 0.1)
        )

    def forward(self, embeddedVector: torch.Tensor) -> torch.Tensor:
        return self.neural_net(embeddedVector)        

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dimension, key_query_reduced_dimensionality, no_of_attention_heads, expansion_factor = 4):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.attention_layer = AttentionLayer(no_of_attention_heads, key_query_reduced_dimensionality, self.embedding_dimension)
        self.normal_1 = nn.LayerNorm(self.embedding_dimension)
        self.mlp_layer = MLPLayer(self.embedding_dimension, expansion_factor)
        self.normal_2 = nn.LayerNorm(self.embedding_dimension)
    
    def forward(self, embeddedVector: torch.Tensor) -> torch.Tensor:
        embeddedVector = embeddedVector + self.attention_layer(self.normal_1(embeddedVector))
        embeddedVector = embeddedVector + self.mlp_layer(self.normal_2(embeddedVector))
        return embeddedVector
    

class GPTTransformer(nn.Module):
    def __init__(self, context_size, no_of_blocks, embedding_dimension, key_query_reduced_dimensionality, no_of_attention_heads, expansion_factor = 4):
        super().__init__()
        self.context_size = context_size
        self.no_of_blocks = no_of_blocks

        #Embedding first
        self.token_embedder = nn.Embedding(vocab_size, embedding_dimension)
        self.position_embedder = nn.Embedding(context_size, embedding_dimension)
        #All transformations
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(embedding_dimension, key_query_reduced_dimensionality, no_of_attention_heads, expansion_factor) for _ in range(no_of_blocks)])

        self.normal = nn.LayerNorm(embedding_dimension)
        self.final_layer = nn.Linear(embedding_dimension, vocab_size)
    
    def forward(self, tokens: torch.Tensor, ideal_value: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, T = tokens.shape
        token_embedding = self.token_embedder(tokens)
        position_embedding = self.position_embedder(torch.arange(T, device = device))

        embeddedVector = token_embedding + position_embedding

        output = self.transformer_blocks(embeddedVector)
        output = self.normal(output)
        output = self.final_layer(output)

        if ideal_value is None:
            loss = None
        else:
            B, T, C = output.shape
            output = output.view(B * T, C)
            ideal_value = ideal_value.view(B * T)

            loss = F.cross_entropy(output, ideal_value)
        
        return output, loss

    def generate(self, tokens: torch.Tensor, max_token_limit: int) -> torch.Tensor:
        for i in range(max_token_limit):
            #Take the last chunk of context
            input_tokens = tokens[:, -self.context_size:]
            output, _ = self(input_tokens)
            #Get the final token which is the newly generated token
            output = output[:, -1, :]
            
            probabilities = F.softmax(output, dim = -1)
            #Pick the most likely token to be picked
            next_token = torch.multinomial(probabilities, num_samples = 1)

            tokens = torch.cat((tokens, next_token), dim = 1)
        return tokens