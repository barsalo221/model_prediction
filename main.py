import os 

import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

def create_labels(base_names, pred_np):
    labels = {}
    for player in base_names:
        player_lines = pred_np[pred_np[:,0]==player]
        if player_lines.shape[0] == 0:
            continue
        
        if player_lines.shape[0]>1:
            player_lines = player_lines[player_lines[:,3]=="TOT"]
            
        player_lines = player_lines[0]
          
        labels[player] = player_lines[-3]
        x = 0
    return labels


def create_ds(base_np, labels, names, teams, pos):

    X = []
    y =[]


    for player in names:
        player_lines = base_np[base_np[:,0]==player]
        if player_lines.shape[0] == 0:
            continue
        if player_lines.shape[0]>1:
            player_lines = player_lines[player_lines[:,3]=="TOT"]
        if player_lines.shape[0] == 0:
            continue
            
        player_lines = player_lines[0]

        data = [ int(np.where(pos==player_lines[1])[0]) ]
        data.append(int(np.where(teams==player_lines[3])[0]) )
        data.append(player_lines[2]/45)
        data.extend([player_lines[4]/82., player_lines[5]/player_lines[4], player_lines[6]/48.])
        data.extend(list(player_lines[7:29]))
        data = np.array(data)

        pt = labels.get(player, None)
        if pt is None:
            continue

        y.append(pt)
        X.append(data)


    return np.array(X), np.array(y)



def main():

    df_base = pd.read_csv(r"nbaPlaterStats22.csv")
    df_pred = pd.read_csv(r"nbaPlaterStats.csv")


    df_base = df_base.drop(["PTSOFTM"], axis=1)
    df_base = df_base.dropna()

    names = np.unique(df_base["Player"])
    teams = np.unique(df_base["Tm"])
    pos = np.unique(df_base["Pos"])

    base_np = np.array(df_base)
    pred_np = np.array(df_pred)

    labels = create_labels(names, pred_np)

    df_base.dropna()

    X, y = create_ds(base_np, labels, names, teams, pos)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    pred = lr.predict(X_test)

    print(metrics.mean_absolute_error(y_test, pred))
    print(metrics.r2_score(y_test, pred))

    with open("teams.txt", "w") as f:
        for team in teams:
            f.write(f"{team}\n")

    with open("pos.txt", "w") as f:
        for pos in pos:
            f.write(f"{pos}\n")

    np.savetxt("coef.txt", lr.coef_)
    np.savetxt("intercept.txt", np.asarray([lr.intercept_]))

    x = .0



if __name__ == "__main__":
    main()