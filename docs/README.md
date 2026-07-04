# Documentation

This directory contains the source files for the project's documentation. The documentation is built using [Sphinx](https://www.sphinx-doc.org/).

## Building the documentation

To build the documentation, you first need to install the project's dependencies:

```
uv pip install -e .[dev]
```

You also need to install Sphinx and the nbsphinx extension:

```
uv pip install sphinx nbsphinx
```

And you need to install pandoc:

```
brew install pandoc
```

Once you have installed the dependencies, you can build the documentation by running the following command from the root of the project:

```
sphinx-quickstart docs
```

Then, you need to configure the `docs/source/conf.py` file to include the `sphinx.ext.autodoc` and `nbsphinx` extensions, and to point to the `src` directory.

Finally, you can build the documentation by running the following command from the `docs` directory:

```
make html
```

The generated HTML documentation will be in the `docs/build/html` directory.
