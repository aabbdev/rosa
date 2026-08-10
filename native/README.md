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

Le module expose aussi `NativeCandidateState` pour l'état riche exact de
`rosa._stateful_candidates_numba`. Son `step` batch maintient les mêmes K
suffixes, R occurrences les plus récentes, fréquences non bornées et tags LCT
`newest-prefix + delta`. La capacité R reste possédée par les tableaux NumPy du
`CandidateState` Python, dont l'objet natif conserve la durée de vie. La
capacité peut être détectée via la présence de `NativeCandidateState` et
`candidate_abi_version == 1`.

Cette première ABI riche expose `step`, `reset` global et `position`. Elle
n'expose pas encore de préremplissage riche ni de reset/continuation masqué par
ligne; ces opérations nécessitent un contrat de positions par ligne distinct
de `CandidateState.position`.

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

`NativeCandidateState(candidate_state).step(tokens_numpy)` attend strictement
un vecteur NumPy C-contigu `int64` et renvoie le tuple bas niveau
`(source, match_length, state_id, frequency, count)`. `reset()` recycle tout le
batch en temps proportionnel aux états et slots de hachage réellement occupés.

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

Le smoke riche et le benchmark direct contre Numba se lancent respectivement
avec `native/tests/candidate_smoke.py` et `native/benchmark_candidates.py` dans
le même environnement isolé.

## Publication multi-plateforme

Étape suivante : ajouter un workflow `cibuildwheel` dédié après avoir fixé la
matrice Python/architectures, les cibles Linux (manylinux) et la politique de
publication. Aucun workflow n'est ajouté ici afin de ne pas publier une
matrice non validée; chaque wheel contient du code natif et doit être produit
séparément par ABI et plateforme.
