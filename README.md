# Prediction of binding free energy

## Dataset description
The dataset is based on [SKEMPI v2.0](https://life.bsc.es/pid/skempi2), atoms from the interaction interface (located at a distance of no more than 4.0 angstroms from the partner protein) were retained from the complex structures and saved in JSON in the following format:
```python
[
    {
        "uid": "1ahw",           # RCSB PDB ID
        "interface_graph": {
            "coords": ...,       # atomic coordinates, N x 3
            "atoms": ...,        # atoms, N
            "residues": ...,     # aminoacids names, N
            "chain_ids": ...,    # chain identifiers, N
            "is_receptor": ...,  # 0 — atom in receptor, 1 — atom in ligand
        },
        "affinity": -12.0        # Free energy of binding
    }
]
```
The samples are located in the `data` folder.


## Environment

Pyenv was used to create the environment because the default _Poetry_ build did not allow for building a working environment. Python 3.10.13 was used for this project. The libraries required to run this project are listed in the `requirements.txt` file in the `requirements` folder.

## Network Training

After creating and activating the virtual environment, you need to train the model. Training is configured via
`train.yaml`.
To do this, you can use the following commands:
```commandline
python -m scripts.train
```
This command will run training on the *GraphNet* model from _modules_. To run training on the more advanced *InvariantGNN* model, you need to add the `--config_name` flag with the 
name of the configuration file (*train_invariant*):
```commandline
python -m scripts.train --config_name train_invariant
```

You can also view the results in TensorBoard. It will look something like this:
!['invariant.png'](./images/invariant.png)
There will also be many other graphs that you can customize.

## Network Testing

After training the network, it needs to be tested. You can test both *GraphNet*
and *InvariantGNN*.

Example of metrics output for the *GraphNet* model.
!['metrics.png'](./images/metrics.png)
Example of metrics output for the *InvariantGNN* model.
!['invariant_pearson.png'](./images/invariant_perason.png)

The launch is similar to the training:
```commandline
python -m scripts.inference
```
```commandline
python -m scripts.inference --config_name train_invariant
```

## Вывод примера

Для вывода примера достаточно через *Jupyter Notebook* запустить файл `interface_graph.ipynb`
и там последовательно запустить чанки для отображения структуры.
!['structure'](./images/cool.png)
## Автор
Беляков Матвей
