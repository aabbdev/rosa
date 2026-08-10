# `rosa-torch-native`

Compagnon natif optionnel de [`rosa-torch`](https://github.com/aabbdev/rosa)
pour l'étape d'inférence exacte SAM + Link-Cut Tree sur CPU. La distribution
installe le module d'extension importable `rosa_native_step`; elle ne remplace
pas le package Python principal.

Le cœur C++ est la copie exacte du prototype validé
`benchmark/production-opt-20260811/native/native_step.cpp`. Il lie une fois les
tableaux NumPy d'un `_StatefulInferenceState`, les modifie sur place et libère
le GIL pendant le calcul. Il n'inclut ni n'appelle libtorch. La dépendance
runtime `rosa-torch[numba]>=0.2,<0.3` fournit le contrat d'état compatible,
PyTorch, NumPy et Numba.

Le constructeur valide intégralement formes, types, compteurs et version ABI
avant de conserver les pointeurs. L'ABI d'état native actuelle vaut `1`.

## Installation et utilisation

Installez un wheel correspondant à la version de Python et à la plateforme :

```bash
python -m pip install rosa-torch-native
```

`rosa-torch` détecte automatiquement le module dans son backend d'inférence
Numba. L'API bas niveau reste disponible pour diagnostic :

```python
from rosa_native_step import NativeState
```

`NativeState(state).step(tokens_numpy)` attend un vecteur NumPy contigu
convertible en `int64`, de forme `[batch_size]`. L'objet conserve une référence
à l'état Python et expose sa `position` en lecture seule.

## Construction locale isolée

Le backend PEP 517 est setuptools, avec pybind11 uniquement comme dépendance de
construction. `Pybind11Extension` sélectionne C++17 et setuptools fournit les
options d'extension propres à macOS/Linux; `-O3` et `NDEBUG` sont ajoutés sur
les deux plateformes.

Depuis la racine du dépôt :

```bash
uv build --wheel native --out-dir /tmp/rosa-native-dist
uv run --isolated \
  --with '.[numba]' \
  --with /tmp/rosa-native-dist/rosa_torch_native-0.2.0-*.whl \
  native/tests/smoke.py
```

Le smoke force Numba comme oracle, laisse le chemin ROSA courant charger le
compagnon, puis compare les prédictions étape par étape.

## Publication multi-plateforme

Étape suivante : ajouter un workflow `cibuildwheel` dédié après avoir fixé la
matrice Python/architectures, les cibles Linux (manylinux) et la politique de
publication. Aucun workflow n'est ajouté ici afin de ne pas publier une
matrice non validée; chaque wheel contient du code natif et doit être produit
séparément par ABI et plateforme.
