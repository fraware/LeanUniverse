# LeanUniverse: A Library for Consistent and Scalable Lean4 Dataset Management
LeanUniverse is a package designed to create comprehensive datasets from Lean4 repositories on Github. Its goal is to simplify and standardize the generation of training datasets for AI models.
The key features include:
- _Consistency_: LeanUniverse ensures that all collected repositories are consistent and can be linked to the same version of dependencies (mathlib). This guarantees reliability and compatibility across datasets created with the library.
- _License Filtering_: Users are empowered to define filters for acceptable licenses and users are responsible for ensuring that their usage of third-party content (github repositories) complies with the associated license and GitHub’s terms of service.
- _Caching_: The library incorporates a caching mechanism, enhancing efficiency by reducing redundant computations. This feature enables recurrent updates and incremental growth of datasets over time.


## Getting Started
LeanUniverse uses [Poetry](https://python-poetry.org/) to manage project dependencies and virtual environments. Follow these steps to get started:

1. Clone the LeanUniverse repository to your local machine:
```
git clone https://github.com/your-repo/LeanUniverse.git
cd LeanUniverse
```
2. Ensure you have Poetry installed for managing dependencies and virtual environments. You can install Poetry using one of the following method:
```
pip install poetry
```
For other installation methods, refer to the [Poetry installation guide](https://python-poetry.org/docs/).
3. Install all required dependencies by running:
```
poetry install
```
4. Activate the environment created by Poetry:
```
poetry shell
```
This sets up a proper shell environment with all dependencies installed.
5. Now, you’re ready to use LeanUniverse! Execute the main script or any specific functionality:
```
python lean_universe/dataset/run.py
```
6. You can add or remove dependencies using poetry.
To add a new dependency:
```
poetry add <package-name>
```
To remove a dependency:
```
poetry remove <package-name>
```
For more information on Poetry and its features, refer to the [official Poetry documentation](https://python-poetry.org/docs/).


## Development
We will be using the using `poetry` to manage the project dependencies and the virtual environments. Once you clone the repo, you should run `potery install`. To run the code you need to run `poetry shell` to get the proper shell environment with everything installed. To add a new dependency you can run `poetry add numpy` and the same way you can remove the dependency.

## License
The model is licensed under the [CC-BY-NC 4.0](LICENSE). Use of this package for commercial purposes is prohibited.

__Important__: Users are responsible for ensuring that their usage of third-party content (github repositories) complies with the associated license and GitHub’s terms of service. LeanUniverse allows filtering of licenses.

## Citation
Please cite as:

``` bibtex
@inproceedings{ott2019fairseq,
  title = {LeanUniverse: A Library for Consistent and Scalable Lean4 Dataset Management},
  author = {Aram H. Markosyan, Gabriel Synnaeve, Hugh Leather},
