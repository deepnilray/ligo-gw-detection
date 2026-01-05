# GitHub Setup Instructions

This guide walks you through setting up the LIGO GW Detection project on GitHub.

## Prerequisites

- GitHub account: https://github.com
- Git installed on your system (download from https://git-scm.com)
- VS Code or command line access

## Step 1: Install Git

If you don't have Git installed:
1. Go to https://git-scm.com/download/win
2. Download and run the installer
3. Accept defaults during installation
4. Restart your terminal after installation

## Step 2: Configure Git Locally

Open PowerShell or Terminal and run:

```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

Replace "Your Name" and "your.email@example.com" with your actual details.

## Step 3: Initialize Local Repository

Navigate to the project directory and initialize git:

```powershell
cd "C:\Users\DEEPNIL RAY\Downloads\ligo-gw-detection"
git init
git add .
git commit -m "Initial commit: LIGO GW detection pipeline with baseline CNN

- Week 1 infrastructure complete: data loaders, transforms, injection generator
- Baseline CNN model (101k parameters) trained on 1000 samples
- Production metrics: AUC=1.0, F1=1.0, Sensitivity=1.0, Specificity=1.0
- Realistic LIGO noise simulation (1/f + glitches)
- Sub-millisecond inference latency
- Publication-ready methods paper"
```

Verify with:
```powershell
git log --oneline
```

## Step 4: Create GitHub Repository

1. Go to https://github.com/new
2. Enter repository name: `ligo-gw-detection`
3. Add description: *Fast Gravitational Wave Detection with Convolutional Neural Networks*
4. Choose visibility: **Public** (to enable open science)
5. **Do NOT** initialize with README, .gitignore, or license (we have these locally)
6. Click "Create repository"

## Step 5: Add Remote and Push

GitHub will show instructions. Follow this:

```powershell
cd "C:\Users\DEEPNIL RAY\Downloads\ligo-gw-detection"

# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/ligo-gw-detection.git

# Verify
git remote -v

# Push to GitHub (use your GitHub token as password if prompted)
git branch -M main
git push -u origin main
```

**For authentication:**
- If prompted for password, use a **Personal Access Token** (not your GitHub password)
- Generate token at: https://github.com/settings/tokens
- Select `repo` scope
- Copy and use as password

## Step 6: Verify on GitHub

Visit: `https://github.com/YOUR_USERNAME/ligo-gw-detection`

You should see:
- ✅ All project files
- ✅ README.md displayed
- ✅ COMMIT_PLAN.md visible
- ✅ papers/ folder with METHODS_PAPER.md
- ✅ ligo_gw/ package
- ✅ scripts/train.py

## Step 7: Add GitHub Badge to README (Optional)

Update [README.md](README.md) with:

```markdown
[![GitHub](https://img.shields.io/badge/GitHub-ligo--gw--detection-blue?logo=github)](https://github.com/YOUR_USERNAME/ligo-gw-detection)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2401.XXXXX-b31b1b)](https://arxiv.org/abs/2401.XXXXX)
```

## Subsequent Commits

After making changes locally:

```powershell
git add .
git commit -m "Descriptive commit message"
git push
```

## Next Steps for Release

- [ ] Add LICENSE file (MIT or Apache 2.0)
- [ ] Create GitHub Release tags
- [ ] Submit to arXiv for publication
- [ ] Link preprint in README
- [ ] Add DOI via Zenodo (https://zenodo.org)
- [ ] Create releases for each week's milestones

## Useful Commands

```powershell
# See status
git status

# View commit history
git log --oneline --graph --all

# See what changed
git diff

# Undo last commit (before push)
git reset --soft HEAD~1

# View branches
git branch -a
```

## Troubleshooting

**Authentication failed:**
- Use Personal Access Token, not password
- Token must have `repo` scope
- Generate new token: https://github.com/settings/tokens

**Remote already exists:**
```powershell
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/ligo-gw-detection.git
```

**Permission denied:**
- Verify SSH key is set up or use HTTPS with token
- Or install Git Credential Manager: https://github.com/git-ecosystem/git-credential-manager

## Support

For detailed Git help:
- Official guide: https://git-scm.com/book/en/v2
- GitHub docs: https://docs.github.com/en
- GitHub help: https://github.community
