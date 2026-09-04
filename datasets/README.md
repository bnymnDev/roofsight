# datasets/

Nothing in here is committed except DVC pointer files (`*.dvc`) and this note.

```sh
dvc pull datasets/v0.1.dvc        # fetch a dataset version from the remote
```

Layout of a version:

```
datasets/v0.1/
  images/               anonymized JPEGs
  images.json           image records (after `roofsight data build`)
  annotations.json      COCO with RoofSight fields (after `roofsight label` + review)
datasets/own/           your own photos + .arkit.json sidecars (input to the build)
datasets/raw/           Mapillary downloads before anonymization; never leaves this machine
```
