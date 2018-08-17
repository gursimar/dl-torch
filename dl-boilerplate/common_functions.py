from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize

def compute_bleu(reference_sentence, predicted_sentence):
    """
    Given a reference sentence, and a predicted sentence, compute the BLEU similary between them.
    """
    reference_tokenized = word_tokenize(reference_sentence.lower())
    predicted_tokenized = word_tokenize(predicted_sentence.lower())
    return sentence_bleu([reference_tokenized], predicted_tokenized)

def cudify(objects, gpu):
    output_objs = tuple()
    for obj in objects:
        output_objs = output_objs + (obj.cuda(gpu), )
    return output_objs

def printWords(output_tokens):
    # output_tokens: batch_size, maxSeqLen
    for idx in range(0,output_tokens.size(0)): # batch
        words = []
        for jdx in range(0, output_tokens.size(1)):
            word_index = output_tokens[idx,jdx]
            word_index = int(word_index.data.numpy())
            words.append(target_lang.index2word[word_index])
        print(words)

def extractSentences(output_tokens, ind2word):
    sentences = []
    for batch_id in range(0, output_tokens.size(0)):
        words = [ind2word[int(i.numpy())] for i in output_tokens[batch_id]]
        filtered_words = []
        for word in words:
            filtered_words.append(word)
            if word == 'EOS':
                break
        sentence = ' '.join(filtered_words)
        sentences.append(sentence)
    return sentences

def compute_bleu(reference_sentence, predicted_sentence):
    """
    Given a reference sentence, and a predicted sentence, compute the BLEU similary between them.
    """
    reference_tokenized = word_tokenize(reference_sentence.lower())
    predicted_tokenized = word_tokenize(predicted_sentence.lower())
    return sentence_bleu([reference_tokenized], predicted_tokenized)

# Compute blue scores
def getBlueScores(groundtruth_sentences, output_sentences):
    blue_scores = []
    for sen_id in range(len(groundtruth_sentences)):
        real_sentence = groundtruth_sentences[sen_id]
        predicted_sentence = output_sentences[sen_id]
        blue_score = compute_bleu(real_sentence, predicted_sentence)
        blue_scores.append(blue_score)
    return blue_scores


## These functions may need modification
def inferBatch(batch, encoder, decoder, embedding_layer_input, embedding_layer_target, params, teacherForcing = False):
    input_batch = batch['input']
    input_batch_len = batch['input_len']
    target_batch = batch['target']
    target_batch_len = batch['target_len']
    batch_size = input_batch.size(0)
    max_seq_len = input_batch.size(1)

    # Embedding layer
    input_batch_embedding = embedding_layer_input(input_batch)
    target_batch_embedding = embedding_layer_target(target_batch)


    #Encoder
    input_batch_len_sorted, input_perm_idx = input_batch_len.sort(0, descending=True)
    _, input_perm_idx_resort = input_perm_idx.sort(0, descending = False)

    input_batch_encoder_hidden = encoder.initHidden(batch_size = batch_size)
    input_batch_embedding = input_batch_embedding.index_select(0, input_perm_idx)
    # input_batch_embedding = input_batch_embedding.transpose(0,1) #seq first
    packed_input_batch_embedding = pack_padded_sequence(input_batch_embedding.transpose(0,1), input_batch_len_sorted.cpu().data.numpy())
    packed_input_batch_encoding, input_batch_encoder_hidden = encoder(packed_input_batch_embedding, input_batch_encoder_hidden)
    output_input_batch_encoding, _ = pad_packed_sequence(packed_input_batch_encoding)
    output_input_batch_encoding = output_input_batch_encoding.transpose(0,1) # make it batch first
    output_input_batch_encoding = output_input_batch_encoding.index_select(0, input_perm_idx_resort)

    if params['rnn_type'] == 'LSTM':
        input_batch_encoder_hidden = (input_batch_encoder_hidden[0].index_select(1, input_perm_idx_resort), 
                                     input_batch_encoder_hidden[1].index_select(1, input_perm_idx_resort))
    else:
        input_batch_encoder_hidden = input_batch_encoder_hidden.index_select(1, input_perm_idx_resort)
    input_batch_encoding = input_batch_encoder_hidden[0].squeeze(0)

    output_tokens = torch.empty(batch_size,params['beamLen'])
    ## Decoder
    sos_tokens = target_batch_embedding.narrow(1,0,1)
    decoder_input = sos_tokens
    decoder_hidden = input_batch_encoder_hidden
    for idx in range(0,params['beamLen']):
        print(decoder_input.size())
        print(decoder_hidden[0].size())
        print(output_input_batch_encoding.size())
        decoder_output, decoder_hidden, _ = decoder(decoder_input.transpose(0,1), decoder_hidden, output_input_batch_encoding, isPacked=False)
        output_probabilities = decoder_output.squeeze(0) # make it batch first
        output_probabilities = F.softmax(output_probabilities, 1)
        values, indices = torch.topk(output_probabilities, 1, dim=1, largest = True, sorted=False)
        decoder_input = embedding_layer_target(indices)
        if teacherForcing: # this is cheating, just to test
            decoder_input = target_batch_embedding[:,idx+1,:].unsqueeze(1)
        output_tokens[:,idx] = indices.squeeze()
        
    return output_tokens

def beamSearch(input_tokens, encoder_batch_hidden, encoder_outputs, embedding_layer, decoder, params, word2ind, ind2word, vocab_size):
    batch_size = input_tokens.size(0)
    sequence_length = params['beamLen']
    beam_size = params['beamSize']
    input_tokens = input_tokens.narrow(1,0,1) # just pick SOS

    input_tokens_embedding = embedding_layer(input_tokens)
#     print(input_tokens_embedding.size())
#     print(encoder_batch_hidden[0].size())
#     print(encoder_outputs.size())
    
    output_probabilities, encoder_batch_hidden, _ = decoder(input_tokens_embedding.transpose(0,1), encoder_batch_hidden, encoder_outputs, isPacked=False)
    if params['rnn_type'] == 'LSTM':
        encoder_batch_hidden = (encoder_batch_hidden[0].view(-1,1,params['hidden_size']), encoder_batch_hidden[1].view(-1,1,params['hidden_size']))
        beam_hidden_state = (encoder_batch_hidden[0].repeat(1,beam_size,1).view(params['num_layers'], -1, params['hidden_size']), encoder_batch_hidden[1].repeat(1,beam_size,1).view(params['num_layers'], -1, params['hidden_size']))
    else:
        encoder_batch_hidden = encoder_batch_hidden.view(-1,1,params['hidden_size'])
        beam_hidden_state = encoder_batch_hidden.repeat(1,beam_size,1).view(params['num_layers'], -1, params['hidden_size'])
    
    encoder_outputs = encoder_outputs.repeat(beam_size,1,1) # added by simar (hacked)
    
    output_probabilities = F.log_softmax(output_probabilities, 2)
    values, indices = torch.topk(output_probabilities, beam_size, dim=2, largest=True, sorted=False)
    sequence_all = Variable(torch.zeros(batch_size, beam_size, sequence_length).long())
    sequence = Variable(torch.zeros(sequence_length, batch_size).long())
    sequence_probabilities = Variable(torch.zeros(sequence_length, batch_size, beam_size).float())
    EOS_TOKEN = word2ind['EOS']
    masked_vector = Variable(torch.zeros(1, vocab_size).float())
    masked_vector = masked_vector - 99999
    masked_vector[0,0] = 0
#     print(masked_vector.size())

    indexer = Variable(torch.arange(batch_size).long().unsqueeze(1).expand_as(indices.squeeze(0))*beam_size)
    masking_batch_num = Variable(torch.arange(batch_size*beam_size).long())
    if params['USE_CUDA']:
        sequence_all = sequence_all.cuda(params['gpu'])
        sequence = sequence.cuda(params['gpu'])
        sequence_probabilities = sequence_probabilities.cuda(params['gpu'])
        indexer = indexer.cuda(params['gpu'])
        masking_batch_num = masking_batch_num.cuda(params['gpu'])
        masked_vector = masked_vector.cuda(params['gpu'])

    sequence_all[:,:,0] = indices
    sequence_probabilities[0] = values
    beam_probability_sum = values.clone().squeeze(0).unsqueeze(2)
    for current_index in range(1, sequence_length):
#         print (current_index)
        #Select next words
        current_input_words = sequence_all[:,:,current_index - 1].clone().view(-1,1)
        mask = sequence_all == EOS_TOKEN
        mask = torch.max(mask, 2)[0].view(-1,)
        lengths = (sequence_all != 0).float()
        lengths = torch.sum(lengths, 2).unsqueeze(2)
        current_input_words_embeddings = embedding_layer(current_input_words)
        current_input_words_embeddings = current_input_words_embeddings.transpose(0,1)

        #Pass through Decoder
#         print(current_input_words_embeddings.size())
#         print(beam_hidden_state[0].size())
#         print(encoder_outputs.size())
        
        current_output_probabilities, beam_hidden_state, _ = decoder(current_input_words_embeddings, beam_hidden_state, encoder_outputs, isPacked=False)
        current_output_probabilities = F.log_softmax(current_output_probabilities, 2).view(-1, current_output_probabilities.size(2))

        #Masking EOS
        masked_indexes = masking_batch_num[mask]
#         print(current_output_probabilities)
        if masked_indexes.nelement() > 0:
            masking_vectors = masked_vector.repeat(masked_indexes.size(0), 1)
            current_output_probabilities.index_copy_(0, masked_indexes, masking_vectors)

        current_output_probabilities = current_output_probabilities.view(batch_size, beam_size, -1)
        # lengths = lengths.expand_as(current_output_probabilities)

        #Update Parameters for next iteration
        current_total_probabilities = beam_probability_sum.expand_as(current_output_probabilities) + current_output_probabilities
        # current_total_probabilities = current_total_probabilities/lengths
        current_values, current_indices = torch.topk(current_total_probabilities.view(batch_size,-1), beam_size, dim=1, largest=True, sorted=False)
        next_indices = current_indices/vocab_size
        next_words = current_indices%vocab_size

        next_indices_adjusted = next_indices + indexer
        next_indices_adjusted = next_indices_adjusted.view(-1,)

        sequence_all = torch.index_select(sequence_all.view(-1, sequence_length), 0, next_indices_adjusted).view(-1, beam_size, sequence_length)
        sequence_all[:,:,current_index] = next_words

        if params['rnn_type'] == 'LSTM':
            beam_hidden_state = (beam_hidden_state[0].transpose(0,1), beam_hidden_state[1].transpose(0,1))
            next_beam_hidden_state = (torch.index_select(beam_hidden_state[0], 0, next_indices_adjusted), torch.index_select(beam_hidden_state[1], 0, next_indices_adjusted))
            next_beam_hidden_state = (next_beam_hidden_state[0].transpose(0,1), next_beam_hidden_state[1].transpose(0,1))
        else:
            beam_hidden_state = beam_hidden_state.transpose(0,1)
            next_beam_hidden_state = torch.index_select(beam_hidden_state, 0, next_indices_adjusted)
            next_beam_hidden_state = next_beam_hidden_state.transpose(0,1)

        beam_probability_sum = current_values.unsqueeze(2)/lengths
        beam_hidden_state = next_beam_hidden_state

    return beam_probability_sum, sequence_all.data.cpu().numpy(), input_tokens.data.cpu().numpy()
#     test_index = 25
#     print (lengths)
#     print (beam_probability_sum[test_index])
#     tokens = sequence_all.data.cpu().numpy()
#     for tok in tokens[test_index]:
#         print (" ".join([ind2word[x] for x in tok if x != 0]))

#     tokens = input_tokens.transpose(0,1).data.cpu().numpy()
#     print(tokens)
#     for tok in tokens[test_index]:
#         print (" ".join([ind2word[x] for x in tok if x != 0]))
#     aa        