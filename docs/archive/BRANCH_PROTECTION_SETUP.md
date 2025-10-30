# branch protection setup

this guide walks through setting up branch protection rules for the main branch to ensure all code changes go through pull requests and ci/cd checks.

## github branch protection rules

### step 1: navigate to settings

1. go to your repository: https://github.com/cjaron03/flare-plus
2. click **settings** tab
3. click **branches** in the left sidebar

### step 2: add branch protection rule

click **add branch protection rule** or **add rule**

### step 3: configure protection rules

**branch name pattern:**
```
main
```

**protect matching branches - enable these:**

#### required checks
- ☑ **require a pull request before merging**
  - ☑ require approvals: 0 (or 1 if you want self-review)
  - ☑ dismiss stale pull request approvals when new commits are pushed
  - ☐ require review from code owners (optional)

- ☑ **require status checks to pass before merging**
  - ☑ require branches to be up to date before merging
  - search and add these status checks:
    - `test (3.9)` - tests on python 3.9
    - `test (3.10)` - tests on python 3.10
    - `test (3.11)` - tests on python 3.11
    - `data-quality-check` - noaa endpoint validation
    - `validate-pr` - pr validation checks
    - `security-scan` - security scanning
    - `code-quality` - code quality checks

#### additional protections
- ☑ **require conversation resolution before merging**
- ☑ **require signed commits** (optional, recommended for security)
- ☑ **require linear history** (optional, keeps history clean)
- ☑ **include administrators** (apply rules to admins too)
- ☐ allow force pushes (keep disabled)
- ☐ allow deletions (keep disabled)

### step 4: save changes

click **create** or **save changes** at the bottom

## development workflow

once branch protection is enabled, you **cannot** push directly to main. here's the proper workflow:

### 1. create a feature branch

```bash
git checkout -b feature/your-feature-name
```

branch naming conventions:
- `feature/` - new features
- `fix/` - bug fixes
- `refactor/` - code refactoring
- `docs/` - documentation updates
- `test/` - test additions/updates

### 2. make your changes

```bash
# make changes to files
git add .
git commit -m "your lowercase commit message"
```

### 3. push branch to github

```bash
git push -u origin feature/your-feature-name
```

### 4. create pull request

1. go to your repository on github
2. click **pull requests** tab
3. click **new pull request**
4. select your branch
5. fill in pr description with:
   - what changes were made
   - why they were made
   - any testing done
6. click **create pull request**

### 5. wait for ci/cd checks

github actions will automatically run:
- unit tests on python 3.9, 3.10, 3.11
- linting (flake8)
- code formatting checks (black)
- security scans
- code quality checks
- noaa endpoint validation

### 6. review and merge

once all checks pass:
1. review the changes
2. click **squash and merge** or **merge pull request**
3. delete the branch after merging

## running checks locally

to catch issues before pushing:

### install pre-commit hooks

```bash
pip install pre-commit
pre-commit install
```

now hooks run automatically on `git commit`.

### run all hooks manually

```bash
pre-commit run --all-files
```

### run tests locally

```bash
# install test dependencies
pip install pytest pytest-cov

# run tests
pytest tests/ -v

# run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

### run linting

```bash
# check code style
flake8 src/

# check formatting
black --check src/

# auto-fix formatting
black src/
```

## emergency access

if you need to push directly to main (emergency only):

1. temporarily disable branch protection in settings
2. make your urgent fix
3. re-enable branch protection immediately
4. create a follow-up pr to properly document the change

## common scenarios

### scenario 1: ci/cd check fails

1. check the failed check in the pr
2. click **details** to see the error
3. fix the issue in your branch
4. commit and push the fix
5. ci/cd automatically re-runs

### scenario 2: merge conflicts

```bash
# update your branch with main
git checkout feature/your-branch
git fetch origin
git merge origin/main

# resolve conflicts
git add .
git commit -m "resolve merge conflicts"
git push
```

### scenario 3: need to update pr after review

```bash
# make requested changes
git add .
git commit -m "address pr feedback"
git push
```

## best practices

1. **keep prs small** - easier to review and test
2. **write descriptive commit messages** - lowercase, concise
3. **test locally first** - run tests and linting before pushing
4. **keep branch updated** - regularly merge main into your branch
5. **delete merged branches** - keeps repo clean
6. **use draft prs** - for work in progress

## verifying protection is active

try pushing directly to main:

```bash
git checkout main
git commit --allow-empty -m "test commit"
git push
```

you should see:
```
! [remote rejected] main -> main (protected branch hook declined)
```

this confirms branch protection is working.

