import math
import torch
import sys
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
from hyperparameters import *
from utils import *

TF = 1

#Encoder-Decoder model with AEL
class Encoder_Decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        self.emb_dim = EMBEDDING_DIM
        self.enc_hidden_dim = HIDDEN_SIZE
        self.dec_hidden_dim = 2*self.enc_hidden_dim

        self.embedding = nn.Embedding(vocab_size, self.emb_dim)
        #Uncomment to use one-hot representation
        # self.embedding.weight.data = torch.eye(vocab_size)
        # self.embedding.weight.requires_grad=False

        self.encoder = nn.LSTM(self.emb_dim, self.enc_hidden_dim, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(self.emb_dim, self.dec_hidden_dim, batch_first=True)
        self.dense = nn.Linear(self.dec_hidden_dim, vocab_size)
        self.dropout = nn.Dropout(DROPOUT)

    def next_activity_prediction(self,sequences,sequence_lengths,targets):
        ''' targets = [batch_size,2]'''

        outputs = []
        ael_outputs = []
        #-----------Encoder--------------------------------------------------
        embedded_inputs = self.embedding(sequences) #[batch,max_len,emb_dim]
        embedded_inputs = nn.utils.rnn.pack_padded_sequence(embedded_inputs, sequence_lengths, batch_first=True)

        #For LSTM
        # _, (hidden,cell)  = self.encoder(embedded_inputs)
        # hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1).unsqueeze(0)
        # cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim = 1).unsqueeze(0)

        #For GRU
        _, hidden  = self.encoder(embedded_inputs)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1).unsqueeze(0)
        #-------------------------------------------------------------------

        GO_embeddings = self.embedding(targets[:,0].unsqueeze(1))

        #For LSTM
        # output, (hidden,cell) = self.decoder(GO_embeddings, (hidden,cell)) #output = [batch_size,1,hidden]

        #For GRU
        output, hidden = self.decoder(GO_embeddings, hidden) #output = [batch_size,1,hidden]

        output = self.dense(output) # [batch_size, 1 , vocab_size]
        outputs.append(output)
        softmax_output = F.softmax(output.squeeze(1),dim=1)
        ael_output = torch.mm(softmax_output,self.embedding.weight).unsqueeze(1)
        ael_outputs.append(ael_output)
        outputs = torch.cat(outputs, dim=1)
        ael_outputs = torch.cat(ael_outputs, dim=1)

        return outputs,ael_outputs
    
    def suffix_prediction(self,sequences,sequence_lengths,targets):
        
        #-----------Encoder--------------------------------------------------
        embedded_inputs = self.embedding(sequences) #[batch,max_len,emb_dim]
        embedded_inputs = nn.utils.rnn.pack_padded_sequence(embedded_inputs, sequence_lengths, batch_first=True)
        
        #For LSTM
        _, (hidden,cell)  = self.encoder(embedded_inputs)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1).unsqueeze(0)
        cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim = 1).unsqueeze(0)
        curr_state = hidden #[1,batch_size,hidden_size]
        curr_cell = cell

        #For GRU
        # _, hidden  = self.encoder(embedded_inputs)
        # hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1).unsqueeze(0)
        # curr_state = hidden #[1,batch_size,hidden_size]
        #-------------------------------------------------------------------
                    
        outputs = []
        ael_outputs = []
        
        next_word_embedding = self.embedding(targets[:,0].unsqueeze(1)) #[batch,1,emb_dim] #Start of sequence tags

        for i in range(1,targets.shape[1]):               
            output, (curr_state,curr_cell) = self.decoder(next_word_embedding, (curr_state,curr_cell)) #output = [batch_size,1,hidden]
            # output, curr_state = self.decoder(next_word_embedding, curr_state) #output = [batch_size,1,hidden]
            output = self.dense(output) # [batch_size, 1 , vocab_size]
            outputs.append(output)
            softmax_output = F.softmax(output.squeeze(1),dim=1)
            next_word_embedding = torch.mm(softmax_output,self.embedding.weight).unsqueeze(1)
            ael_outputs.append(next_word_embedding)
            
            teacher_forcing = random.random() < TF
            if teacher_forcing :
                next_word_embedding = self.embedding(targets[:,i].unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        ael_outputs = torch.cat(ael_outputs, dim=1)

        return outputs,ael_outputs

    def infer(self,sequences,sequence_lengths,targets):
        #-----------Encoder--------------------------------------------------
        embedded_inputs = self.embedding(sequences) #[batch,max_len,emb_dim]
        embedded_inputs = nn.utils.rnn.pack_padded_sequence(embedded_inputs, sequence_lengths, batch_first=True)
        
        #For LSTM
        _, (hidden,cell)  = self.encoder(embedded_inputs)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1).unsqueeze(0)
        cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim = 1).unsqueeze(0)
        curr_state = hidden #[1,batch_size,hidden_size]
        curr_cell = cell

        #For GRU
        # _, hidden  = self.encoder(embedded_inputs)
        # hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1).unsqueeze(0)
        # curr_state = hidden #[1,batch_size,hidden_size]
        #-------------------------------------------------------------------

        predictions = []
        next_word_embedding = self.embedding(targets[:,0].unsqueeze(1)) #[batch,1,emb_dim]

        counter = 0
        while True:
            counter += 1
            output, (curr_state,curr_cell) = self.decoder(next_word_embedding, (curr_state,curr_cell)) #output = [batch_size,1,hidden]
            # output, curr_state = self.decoder(next_word_embedding, curr_state) #output = [batch_size,1,hidden]
            output = self.dense(output) # [batch_size, 1 , vocab_size]
            softmax_output = F.softmax(output.squeeze(1),dim=1)
            pred_var = torch.max(softmax_output,dim=-1,keepdim=True)[1] #[batch_size,1]
            predictions.append(pred_var.item())
            if pred_var.item() == EOS_ID or len(predictions) > MAX_SUFX_LENGTH :
                break
            next_word_embedding = torch.mm(softmax_output,self.embedding.weight).unsqueeze(1)

        return predictions

    def forward(self, sequences, sequence_lengths, targets,task): 
        ''' sequences = [batch_size,max_len1], sequence_lengths = [batch_size]
            targets   = [batch_size,max_len2] '''

        if task =="next_activity_prediction":
           outputs,ael_outputs = self.next_activity_prediction(sequences,sequence_lengths,targets[:,:2])

        elif task == "suffix_prediction":
            outputs,ael_outputs = self.suffix_prediction(sequences,sequence_lengths,targets)

        
        return outputs,ael_outputs

#Conditional VAE model with AEL
class Old_VarGenerator(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        self.emb_dim = EMBEDDING_DIM
        self.enc_hidden_dim = HIDDEN_SIZE
        self.dec_hidden_dim = 2*self.enc_hidden_dim

        self.embedding = nn.Embedding(vocab_size, self.emb_dim)
        #Uncomment to use one-hot representation
        self.embedding.weight.data = torch.eye(vocab_size)
        self.embedding.weight.requires_grad=False

        self.hidden_to_mu = nn.Linear(self.dec_hidden_dim,self.enc_hidden_dim)
        self.hidden_to_logsigma = nn.Linear(self.dec_hidden_dim,self.enc_hidden_dim)
        self.activation_function = nn.ReLU()

        self.encoder = nn.GRU(self.emb_dim, self.enc_hidden_dim, batch_first=True, bidirectional=True)
        self.decoder = nn.GRU(self.emb_dim+self.enc_hidden_dim, self.dec_hidden_dim, batch_first=True)
        self.dense = nn.Linear(self.dec_hidden_dim, vocab_size)
        self.dropout = nn.Dropout(DROPOUT)

    def reparameterize(self,state):
        hidden = state
        mu = self.hidden_to_mu(hidden)
        logsigma = self.hidden_to_logsigma(hidden)
        std = logsigma.mul(0.5).exp_()

        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return z, mu, logsigma

    def compute_kld(self, mu, logsigma):
        kld = -0.5 * torch.sum(1+ logsigma - mu.pow(2) - logsigma.exp())
        return kld
            
    def forward(self, sequences, sequence_lengths, targets, target_lengths,tf_ratio=0.5):
        ''' sequences = [batch_size,max_len1], sequence_lengths = [batch_size]
            targets   = [batch_size,max_len2], target_lengths = [batch_size] 
            tf_ratio = TeacherForcing Ratio'''


        #---------Encoder----------------------------------------------------
        embedded_inputs = self.embedding(sequences) #[batch,max_len,emb_dim]
        embedded_inputs = nn.utils.rnn.pack_padded_sequence(embedded_inputs, sequence_lengths, batch_first=True)
        _, hidden  = self.encoder(embedded_inputs)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1).unsqueeze(0)
        #--------------------------------------------------------------------
        
        outputs = []
        ael_outputs  = []
        curr_state = hidden #[1,batch_size,dec_hid_dim]
        next_word_embedding = self.embedding(targets[:,0].unsqueeze(1)) #[batch,1,emb_dim] #GO_ID tags
        kld = 0

        for i in range(1,targets.shape[1]):
          
            z, mu, logsigma = self.reparameterize(curr_state.squeeze(0)) # z,mu,logsigma=[batch,enc_hid_dim]
            kld += self.compute_kld(mu, logsigma)
            decoder_input = torch.cat([next_word_embedding, z.unsqueeze(1)], -1)
            output, curr_state = self.decoder(decoder_input, curr_state) #output = [batch_size,1,dec_hid_dim]
            output = self.dropout(output)
            predicted_output = self.dense(output) #[batch,1,vocab_size]
            outputs.append(predicted_output)

            teacher_forcing = random.random() < tf_ratio
            if teacher_forcing:
                next_word_embedding = self.embedding(targets[:,i].unsqueeze(1)) #[batch,1,emb_dim]
            else :
                softmax_output = F.log_softmax(predicted_output.squeeze(1),dim=-1)
                # dec_inp_var = torch.max(softmax_output,dim=-1,keepdim=True)[1] #[batch_size,1]
                next_word_embedding = torch.mm(softmax_output,self.embedding.weight).unsqueeze(1)
                ael_outputs.append(next_word_embedding)
                # next_word_embedding = self.embedding(dec_inp_var)

        outputs = torch.cat(outputs, dim=1)
        ael_outputs = torch.cat(ael_outputs,dim=1)

        return outputs,kld,ael_outputs
        
#New VAE
class New_VarGenerator(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        self.emb_dim = vocab_size
        self.enc_hidden_dim = HIDDEN_SIZE
        self.dec_hidden_dim = 2*self.enc_hidden_dim

        self.embedding = nn.Embedding(vocab_size, self.emb_dim)
        #Uncomment to use one-hot representation
        self.embedding.weight.data = torch.eye(vocab_size)
        self.embedding.weight.requires_grad=False

        self.hidden_to_mu = nn.Linear(self.dec_hidden_dim,self.dec_hidden_dim)
        self.hidden_to_logsigma = nn.Linear(self.dec_hidden_dim,self.dec_hidden_dim)
        self.activation_function = nn.ReLU()

        self.encoder = nn.GRU(self.emb_dim, self.enc_hidden_dim, batch_first=True, bidirectional=True)
        self.decoder = nn.GRU(self.emb_dim, self.dec_hidden_dim, batch_first=True)
        self.dense = nn.Linear(self.dec_hidden_dim, vocab_size)
        self.dropout = nn.Dropout(DROPOUT)

    def reparameterize(self,state):
        hidden = state
        mu = self.hidden_to_mu(hidden)
        logsigma = self.hidden_to_logsigma(hidden)
        std = logsigma.mul(0.5).exp_()

        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return z, mu, logsigma

    def compute_kld(self, mu, logsigma):
        kld = -0.5 * torch.sum(1+ logsigma - mu.pow(2) - logsigma.exp())
        return kld

    def next_activity_prediction(self,sequences,sequence_lengths,targets):
        ''' targets = [batch_size,2]'''

        outputs = []
        ael_outputs = []
        #-----------Encoder--------------------------------------------------
        embedded_inputs = self.embedding(sequences) #[batch,max_len,emb_dim]
        embedded_inputs = nn.utils.rnn.pack_padded_sequence(embedded_inputs, sequence_lengths, batch_first=True)

        #For LSTM
        # _, (hidden,cell)  = self.encoder(embedded_inputs)
        # hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1).unsqueeze(0)
        # cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim = 1).unsqueeze(0)

        #For GRU
        _, hidden  = self.encoder(embedded_inputs)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1).unsqueeze(0)
        #-------------------------------------------------------------------

        z, mu, logsigma = self.reparameterize(hidden.squeeze(0)) # z,mu,logsigma=[batch,enc_hid_dim]
        kld = self.compute_kld(mu, logsigma)

        GO_embeddings = self.embedding(targets[:,0].unsqueeze(1))
        
        #For LSTM
        # output, (hidden,cell) = self.decoder(GO_embeddings, (z.unsqueeze(0),cell)) #output = [batch_size,1,hidden]

        #For GRU
        output, hidden = self.decoder(GO_embeddings, z.unsqueeze(0)) #output = [batch_size,1,hidden]

        output = self.dense(output) # [batch_size, 1 , vocab_size]
        outputs.append(output)
        softmax_output = F.softmax(output.squeeze(1),dim=1)
        ael_output = torch.mm(softmax_output,self.embedding.weight).unsqueeze(1)
        ael_outputs.append(ael_output)
        outputs = torch.cat(outputs, dim=1)
        ael_outputs = torch.cat(ael_outputs, dim=1)

        return outputs,kld,ael_outputs
            
    def suffix_prediction(self,sequences,sequence_lengths,targets):
        #---------Encoder----------------------------------------------------
        embedded_inputs = self.embedding(sequences) #[batch,max_len,emb_dim]
        embedded_inputs = nn.utils.rnn.pack_padded_sequence(embedded_inputs, sequence_lengths, batch_first=True)
        _, hidden  = self.encoder(embedded_inputs)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1).unsqueeze(0)
        #--------------------------------------------------------------------
        
        outputs = []
        ael_outputs  = []
        curr_state = hidden #[1,batch_size,dec_hid_dim]
        z, mu, logsigma = self.reparameterize(curr_state.squeeze(0)) # z,mu,logsigma=[batch,enc_hid_dim]
        kld = self.compute_kld(mu, logsigma)
        next_word_embedding = self.embedding(targets[:,0].unsqueeze(1)) #[batch,1,emb_dim] #GO_ID tags

        for i in range(1,targets.shape[1]):
          
            
            decoder_input = torch.cat([next_word_embedding, z.unsqueeze(1)], -1)
            output, curr_state = self.decoder(decoder_input, curr_state) #output = [batch_size,1,dec_hid_dim]
            output = self.dropout(output)
            predicted_output = self.dense(output) #[batch,1,vocab_size]
            outputs.append(predicted_output)

            teacher_forcing = random.random() < tf_ratio
            if teacher_forcing:
                next_word_embedding = self.embedding(targets[:,i].unsqueeze(1)) #[batch,1,emb_dim]
            else :
                softmax_output = F.log_softmax(predicted_output.squeeze(1),dim=-1)
                # dec_inp_var = torch.max(softmax_output,dim=-1,keepdim=True)[1] #[batch_size,1]
                next_word_embedding = torch.mm(softmax_output,self.embedding.weight).unsqueeze(1)
                ael_outputs.append(next_word_embedding)
                # next_word_embedding = self.embedding(dec_inp_var)

        outputs = torch.cat(outputs, dim=1)
        ael_outputs = torch.cat(ael_outputs,dim=1)

        return outputs,kld,ael_outputs

    def forward(self, sequences, sequence_lengths, targets,task):
        ''' sequences = [batch_size,max_len1], sequence_lengths = [batch_size]
            targets   = [batch_size,max_len2]'''

        if task =="next_activity_prediction":
           outputs,kld,ael_outputs = self.next_activity_prediction(sequences,sequence_lengths,targets[:,:2])

        elif task == "suffix_prediction":
            outputs,kld,ael_outputs = self.suffix_prediction(sequences,sequence_lengths,targets)

        return outputs,kld,ael_outputs

#Encoder-Decoder model with AEL and time (Not suported)
class Encoder_Decoder_time(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        self.emb_dim = EMBEDDING_DIM
        self.enc_hidden_dim = HIDDEN_SIZE
        self.dec_hidden_dim = 2*self.enc_hidden_dim

        self.embedding = nn.Embedding(vocab_size, self.emb_dim)
        self.embedding.weight.data = torch.eye(vocab_size)
        self.embedding.weight.requires_grad=False

        self.encoder = nn.LSTM(self.emb_dim+1, self.enc_hidden_dim, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(self.emb_dim+1, self.dec_hidden_dim, batch_first=True)
        self.event_dense = nn.Linear(self.dec_hidden_dim, vocab_size)
        self.time_dense = nn.Linear(self.dec_hidden_dim, 1)
        self.dropout = nn.Dropout(DROPOUT)
            
    def forward(self, sequences, sequence_lengths, targets, target_lengths,input_times,target_times,tf_ratio=0.5,use_AEL=False):
        ''' sequences = [batch_size,max_len1], sequence_lengths = [batch_size]
            targets   = [batch_size,max_len2], target_lengths = [batch_size] '''

        embedded_inputs = self.embedding(sequences) #[batch,max_len,emb_dim]
        input_times = input_times.unsqueeze(-1).float() #[batch,max_len,1]
       
        timed_embedded_inputs = torch.cat((embedded_inputs,input_times),dim=-1) #[batch,max_len,emb_dim+1]
        
        embedded_inputs = nn.utils.rnn.pack_padded_sequence(timed_embedded_inputs, sequence_lengths, batch_first=True)
        _, (hidden,cell)  = self.encoder(embedded_inputs)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1).unsqueeze(0)
        cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim = 1).unsqueeze(0)

        event_outputs = []
        time_outputs = []
        ael_embeddings = []
        curr_state = hidden #[1,batch_size,hidden_size]
        curr_cell = cell

        target_times = target_times.unsqueeze(-1).float()
        next_word_embedding = self.embedding(targets[:,0].unsqueeze(1)) #[batch,1,emb_dim]
        timed_next_word = torch.cat((next_word_embedding,target_times[:,0].unsqueeze(-1)),dim=-1) #[batch,1,emb+1]

        for i in range(1,targets.shape[1]):               
            output, (curr_state,curr_cell) = self.decoder(timed_next_word, (curr_state,curr_cell)) #output = [batch_size,1,hidden]
            event_output = self.event_dense(output) # [batch_size, 1 , vocab_size]
            time_output = self.time_dense(output) # [batch_size, 1 , 1]
            event_outputs.append(event_output)
            time_outputs.append(time_output)
            softmax_output = F.softmax(event_output.squeeze(1),dim=1)
            dec_inp_var = torch.max(softmax_output,dim=1,keepdim=True)[1] #[batch_size,1]

            teacher_forcing = random.random() < tf_ratio
            if teacher_forcing :
                next_word_embedding = self.embedding(targets[:,i].unsqueeze(1))
                timed_next_word = torch.cat((next_word_embedding,target_times[:,i].unsqueeze(-1)),dim=-1)
            else :
                if use_AEL:
                    # next_word_embedding = torch.mm(softmax_output,self.embedding.weight).unsqueeze(1)
                    next_word_embedding = softmax_output.unsqueeze(1)
                    timed_next_word = torch.cat((next_word_embedding,time_output),dim=-1)
                    ael_embeddings.append(next_word_embedding)

                else:
                    next_word_embedding = self.embedding(dec_inp_var)
                    timed_next_word = torch.cat((next_word_embedding,time_output),dim=-1)

        event_outputs = torch.cat(event_outputs, dim=1)
        time_outputs = torch.cat(time_outputs,dim=1)

        if use_AEL:
            ael_embeddings = torch.cat(ael_embeddings, dim=1)
        return event_outputs,time_outputs,ael_embeddings

#GAN model storing the Generator and Discriminator
class GAN_AEL(nn.Module):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

class Discriminator_CNN(nn.Module):
    def __init__(self, filter_num, filter_sizes, dropout_p=0.2):
        super().__init__()
        self.emb_dim = EMBEDDING_DIM
        self.query_CNN    = Sequence_CNN(self.emb_dim, filter_num, filter_sizes)
        self.response_CNN = Sequence_CNN(self.emb_dim, filter_num, filter_sizes)
        self.classifier   = nn.Sequential(
                        nn.Linear(2*filter_num*len(filter_sizes), 32),
                        nn.ReLU(),
                        nn.Dropout(p=dropout_p),
                        nn.Linear(32, 1)
                        # nn.Sigmoid()
                    )

    def forward(self, query, response):
        """
        Args:
            - **query** [B, max_len, emb_dim]
            - **response** [B, max_len, emb_dim]
        Output:
            The probability of real
        """
        query_features = self.query_CNN(query) #[batch, q_features(256)]
        response_features = self.response_CNN(response) #[batch, r_features(256)]
        inputs = torch.cat((query_features, response_features), 1) #[batch, all_features(512)]
        prediction = self.classifier(inputs)
        return prediction,response_features

class Sequence_CNN(nn.Module):
    def __init__(self, input_dim, filter_num, filter_sizes):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(1, filter_num, (width, input_dim)) for width in filter_sizes])
        # self.conv = nn.Conv2d(1,filter_num,(5,input_dim))

    def forward(self, embedded_inputs):
        x = embedded_inputs.unsqueeze(1)  # (N, 1, W, D)
        # exit(0)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(B, filter_num), ...]*len(Ks)
        return torch.cat(x, 1) # [B, all_of_features]

#LSTM discriminator model (Not supported)
class Discriminator_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_dim = EMBEDDING_DIM
        self.hidden_dim = HIDDEN_SIZE
        self.query_rnn = nn.LSTM(self.emb_dim, self.hidden_dim, batch_first=True)
        self.response_rnn = nn.LSTM(self.emb_dim, self.hidden_dim, batch_first=True)
        self.dense = nn.Linear(2*self.hidden_dim, 50)
        self.activation_function = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.prob_layer = nn.Linear(50,1)

    def forward(self,query,response):
        #query->[batch,maxlen1,emb], response -> [batch,malen2,emb]
        _,(query_hidden,_) = self.query_rnn(query)
        _,(response_hidden,_) = self.response_rnn(response)

        input_features1 = torch.cat((query_hidden.squeeze(0),response_hidden.squeeze(0)),dim=1)
        input_features2 = self.dropout(self.activation_function(self.dense(input_features1)))
        output = torch.sigmoid(self.prob_layer(input_features2))
        return output,response_hidden.squeeze(0)


