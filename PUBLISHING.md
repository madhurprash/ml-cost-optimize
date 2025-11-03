# Publishing to PyPI

This guide walks you through publishing the `ml-cost-optimize` package to PyPI.

## Prerequisites

1. **PyPI Account**: Create accounts on both:
   - [PyPI](https://pypi.org/account/register/) - for production releases
   - [TestPyPI](https://test.pypi.org/account/register/) - for testing

2. **API Tokens**: Generate API tokens for authentication:
   - Go to Account Settings → API tokens
   - Create a token with appropriate scope
   - Save the token securely (you'll only see it once)

3. **Install Build Tools**:
   ```bash
   pip install --upgrade build twine
   ```

## Step 1: Update Package Metadata

Before publishing, update the following in `pyproject.toml`:

```toml
[project]
name = "ml-cost-optimize"
version = "0.1.0"  # Update version for each release
authors = [
    {name = "Your Name", email = "your.email@example.com"}  # Update with your info
]

[project.urls]
Homepage = "https://github.com/yourusername/ml-cost-optimize"  # Update URLs
Repository = "https://github.com/yourusername/ml-cost-optimize"
Issues = "https://github.com/yourusername/ml-cost-optimize/issues"
```

## Step 2: Clean Previous Builds

Remove any previous build artifacts:

```bash
rm -rf dist/ build/ *.egg-info
```

## Step 3: Build the Package

Build both source distribution and wheel:

```bash
python -m build
```

This creates:
- `dist/ml-cost-optimize-0.1.0.tar.gz` - source distribution
- `dist/ml-cost-optimize-0.1.0-py3-none-any.whl` - wheel distribution

## Step 4: Test on TestPyPI (Recommended)

Before publishing to the real PyPI, test on TestPyPI:

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# You'll be prompted for:
# Username: __token__
# Password: <your-testpypi-token>
```

Test the installation:

```bash
# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple ml-cost-optimize

# Test the CLI
ml-cost-optimize --help
```

**Note**: The `--extra-index-url` is needed because dependencies are on regular PyPI.

## Step 5: Publish to PyPI

Once testing is successful, publish to the real PyPI:

```bash
python -m twine upload dist/*

# You'll be prompted for:
# Username: __token__
# Password: <your-pypi-token>
```

## Step 6: Verify Installation

Test that users can install your package:

```bash
pip install ml-cost-optimize
ml-cost-optimize --help
```

## Using API Tokens with .pypirc

To avoid entering credentials each time, create `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...  # Your PyPI token

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-AgENdGVzdC5weXBp...  # Your TestPyPI token
```

**Security**: Set appropriate permissions:
```bash
chmod 600 ~/.pypirc
```

## Publishing Updates

When releasing a new version:

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "0.1.1"  # Increment version
   ```

2. **Clean and rebuild**:
   ```bash
   rm -rf dist/ build/ *.egg-info
   python -m build
   ```

3. **Upload new version**:
   ```bash
   python -m twine upload dist/*
   ```

## Automation with GitHub Actions

Create `.github/workflows/publish.yml` for automated releases:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

Add your PyPI token as a GitHub secret named `PYPI_API_TOKEN`.

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.0.0): Breaking changes
- **MINOR** (0.1.0): New features, backwards compatible
- **PATCH** (0.1.1): Bug fixes

## Pre-release Checklist

Before publishing:

- [ ] Update version in `pyproject.toml`
- [ ] Update `CHANGELOG.md` with changes
- [ ] Run tests: `pytest`
- [ ] Run linters: `ruff check . && ruff format .`
- [ ] Update README if needed
- [ ] Clean build artifacts
- [ ] Build package: `python -m build`
- [ ] Test on TestPyPI
- [ ] Create git tag: `git tag v0.1.0`
- [ ] Push tag: `git push origin v0.1.0`
- [ ] Publish to PyPI

## Troubleshooting

### Error: File already exists

If you get "File already exists" error:
- You cannot overwrite an existing version on PyPI
- Increment the version number in `pyproject.toml`
- Rebuild and upload again

### Error: Invalid package name

- Package name must be unique on PyPI
- Check if name is available: https://pypi.org/project/ml-cost-optimize/
- If taken, choose a different name in `pyproject.toml`

### Missing files in package

If files are missing after installation:
- Check `MANIFEST.in` includes all necessary files
- Verify `[tool.setuptools.package-data]` in `pyproject.toml`
- Rebuild and test locally: `pip install dist/*.whl`

## Quick Reference

```bash
# Complete publishing workflow
rm -rf dist/ build/ *.egg-info
python -m build
python -m twine upload --repository testpypi dist/*  # Test first
python -m twine upload dist/*  # Publish to PyPI
```

## Resources

- [PyPI](https://pypi.org/)
- [TestPyPI](https://test.pypi.org/)
- [Packaging Tutorial](https://packaging.python.org/tutorials/packaging-projects/)
- [Twine Documentation](https://twine.readthedocs.io/)
