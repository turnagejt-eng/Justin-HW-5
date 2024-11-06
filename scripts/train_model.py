import argparse
import os
import rdkit
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import numpy as np
from rdkit.Chem.AllChem import GetMorganGenerator
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def train_model(radius, n_bits):
    # Load Lipophilicity dataset
    path = "Lipophilicity.csv"
    print(os.path.exists(path))
    Lip_data = pd.read_csv(path)
    
    train_data, test_data = train_test_split(Lip_data, test_size=0.2)
    mfgen = GetMorganGenerator(radius=3, fpSize=2048)

    train_morg_fp = []
    for s in train_data["smiles"]:
        mol = rdkit.Chem.MolFromSmiles(s)
        fp = mfgen.GetFingerprint(mol)
        train_morg_fp.append(fp)
    train_morg_fp = np.array(train_morg_fp)
    train_targets = train_data["exp"]
    print(train_morg_fp.shape)

    test_morg_fp = []
    for s in test_data["smiles"]:
        mol = rdkit.Chem.MolFromSmiles(s)
        fp = mfgen.GetFingerprint(mol)
        test_morg_fp.append(fp)
    test_morg_fp = np.array(test_morg_fp)
    test_targets = test_data["exp"]
    test_morg_fp.shape

    y_train = np.array(train_data['exp'].values)
    y_test = np.array(test_data['exp'].values)

    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    mlp_morgan = MLPRegressor(max_iter=1000)
    mlp_morgan.fit(train_morg_fp, y_train_scaled)
    
    y_pred_morgan = scaler.inverse_transform(mlp_morgan.predict(test_morg_fp).reshape(-1, 1)).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_morgan))

    # Save results
    env_name = os.getenv("CONDA_DEFAULT_ENV")
    with open("results.txt", "w") as f:
        f.write(f"RMSE: {rmse}\nEnvironment: {env_name}\nRadius: {radius}\nN_bits: {n_bits}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", type=int, default=2, help="Radius for Morgan Fingerprints")
    parser.add_argument("--n_bits", type=int, default=1024, help="Number of bits for Morgan Fingerprints")
    args = parser.parse_args()

    train_model(args.radius, args.n_bits)