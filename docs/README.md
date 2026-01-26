# SPEC4py documentation

## Generating the documentation

To generate the SPEC4py documentation you'll first need to install the dependencies:

```bash
pip install -e .[docs]
```

This will install SPEC4py as well as all the dependencies for the documentation. Note the `-e` option to make the installation editable.

Once all the dependencies are installed, you only need to go into the `docs` directory and execute `make` with the appropriate rule corresponding to the format you would like to generate (HTML, LaTeX, *etc.*), *e.g.*:

```bash
make html
```