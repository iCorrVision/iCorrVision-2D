# Data included in the repository

The DIC Challenge images are not included; see the README at the repository root for where the
notebooks expect them. The two files below are included because the results of the notebooks
depend on them.

| File | Used by | Provenance |
| --- | --- | --- |
| `sample12/roi.tiff` | `validation/notebooks/sample12_vsg_study.ipynb` | Region of interest for DIC Challenge 1.0 Sample 12, drawn by the author: the specimen with its central hole excluded. White marks the region of interest. |
| `star56/participant_mei_frozen.csv` | `validation/notebooks/star56_mei.ipynb` | Code MEI of each participant of the DIC Challenge 2.0 Stars 5 and 6, computed by that notebook from the participants' submitted line cuts and frozen once the participant results had been checked against the published figures. |

`participant_mei_frozen.csv` has one row per participant code and quantity, with the columns
`quantity` (`displacement` or `strain`), `code` and `code_mei`. It is the reference of the
notebook's regression check: a change of more than 1 % in any participant's value is reported.
It is derived from the participant results of the DIC Challenge 2.0:

P. L. Reu et al., "DIC Challenge 2.0: Developing Images and Guidelines for Evaluating Accuracy and
Resolution of 2D Analyses", *Experimental Mechanics* 62 (2022) 639–654.
<https://doi.org/10.1007/s11340-021-00806-6>
