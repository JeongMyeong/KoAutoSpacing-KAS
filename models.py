from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def KAS(n_unit, activation):
    input_data_1 = Input(shape=(100,))

    x = Embedding(len(syllable_tokenizer.word_index)+1, 128)(input_data_1)
    conv_1 = Conv1D(n_unit, 2, padding='same' , activation=activation)(x)
    conv_2 = Conv1D(n_unit, 3, padding='same' , activation=activation)(x)
    conv_3 = Conv1D(n_unit, 4, padding='same' , activation=activation)(x)
    conv_4 = Conv1D(n_unit, 5, padding='same' , activation=activation)(x)
    concat = concatenate([conv_1, conv_2, conv_3, conv_4])
    BN = BatchNormalization()(concat)
    
    bidirectional_lstm = Bidirectional(LSTM(128, dropout=0.3, return_sequences=True))(BN)
    lstm_single = LSTM(64, dropout=0.15, return_sequences=True)(bidirectional_lstm)
    
    LN = LayerNormalization()(lstm_single)
    x = (Dense(64, activation='elu'))(LN)
    x = Dropout(0.5)(x)

    
    
    input_data_pos = Input(shape=(100,))
    x_pos = Embedding(len(postag_dic)*2+1, 64)(input_data_pos)
    
    conv_1_pos = Conv1D(n_unit, 2, padding='same' , activation=activation)(x_pos)
    conv_2_pos = Conv1D(n_unit, 3, padding='same' , activation=activation)(x_pos)
    conv_3_pos = Conv1D(n_unit, 4, padding='same' , activation=activation)(x_pos)
    conv_4_pos = Conv1D(n_unit, 5, padding='same' , activation=activation)(x_pos)
    
    concat_pos = concatenate([conv_1_pos, conv_2_pos, conv_3_pos, conv_4_pos])
    BN_pos = BatchNormalization()(concat_pos)

    bidirectional_lstm = Bidirectional(LSTM(128, dropout=0.3, return_sequences=True))(BN_pos)

    lstm_pos = LSTM(64, dropout=0.3, return_sequences=True)(bidirectional_lstm)
    LN_pos = LayerNormalization()(lstm_pos)
    x_pos = (Dense(64, activation='relu'))(LN_pos)
    x_pos = Dropout(0.5)(x_pos)


    concat_tag_pos = concatenate([x, x_pos])


    x = (Dense(64, activation=activation))(concat_tag_pos)
    output = (Dense(3, activation='softmax'))(x)
    model_sm = Model([input_data_1,input_data_pos], output)
    return model_sm