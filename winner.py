import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import keras
from keras.models import Model
from keras.layers import Input,Dense

def get_winnerList_ttest(x, y, alpha):
    case = np.argwhere(y == 1)[:, 0]
    control = np.argwhere(y == 0)[:, 0]
    x_case, x_control = x[case], x[control]
    p_value = [ttest_ind(x_case[:,i], x_control[:,i]).pvalue for i in range(x.shape[1])]
    winner_list = np.argwhere(np.array(p_value) <= alpha)[:,0]
    return winner_list

def predict_leukmia_attention(x_train, x_test, y_train, y_test):
    input_vector = Input(shape=(x_train.shape[1],))
    attention_probs = Dense(x_train.shape[1], activation='softmax')(input_vector) # similarity
    attention_mul = keras.layers.multiply([input_vector, attention_probs]) # weight
    dense = Dense(85, activation='relu')(attention_mul)
    dense = Dense(85, activation='relu')(dense)
    output_class = Dense(2, activation='sigmoid')(dense)

    model = Model(input_vector, output_class)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model.fit(x_train, y_train, batch_size=32, epochs=32, verbose=1, validation_split=0.1)
    model.summary()

    attention_mul = Model(input_vector, attention_mul)
    x_train_att = attention_mul.predict(x_train)
    x_test_att = attention_mul.predict(x_test)
    return x_train_att, x_test_att

if __name__ == "__main__":
    df = pd.read_csv("leukdata.csv", header=None)
    x = np.transpose(df.iloc[1:,1:].values)
    y = np.ravel(df.iloc[0, 1:].values)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.31, random_state=1)

    # ------ all features to predict leukmia ------
    RF_all = RandomForestClassifier(random_state=1)
    RF_all.fit(x_train, y_train)
    y_pred_all = RF_all.predict(x_test)
    # print("## all features:", accuracy_score(y_test, y_pred))

    # ------ features of winner list to predict leukmia ------
    winnerIndex = get_winnerList_ttest(x, y, 0.001)
    # print("## winner list:", winnerIndex[0:10])
    # print("## number of gene:", len(winnerIndex))
    x_winner_train, x_winner_test, y_winner_train, y_winner_test = train_test_split(x[:, winnerIndex], y, test_size=0.31, random_state=1)
    RF_winner = RandomForestClassifier(random_state=1)
    RF_winner.fit(x_winner_train, y_winner_train)
    y_pred_winner = RF_winner.predict(x_winner_test)
    # print("## features of winner list:", accuracy_score(y_winner_test, y_winner_pred))

    # ------ predict leukmia using attention ------
    y_onehot = OneHotEncoder().fit_transform(y.reshape(-1,1)).toarray()
    x_train, x_test, y_onehot_train, y_onehot_test = train_test_split(x, y_onehot, test_size=0.31, random_state=1)
    x_train_att, x_test_att = predict_leukmia_attention(x_train, x_test, y_onehot_train, y_onehot_test)    
    RF_attention = RandomForestClassifier(random_state=1)
    RF_attention.fit(x_train_att, y_train)
    y_pred_att = RF_attention.predict(x_test_att)

    print("## all features:", accuracy_score(y_test, y_pred_all))
    print("## features of winner list:", accuracy_score(y_test, y_pred_winner))
    print("## features brief from attention:", accuracy_score(y_test, y_pred_att))