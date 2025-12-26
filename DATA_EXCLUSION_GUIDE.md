# Data Exclusion Guide - NutriSenseAI

## ✅ Your Data is Protected from GitHub Upload

**Date:** December 26, 2025

---

## Summary

Your `.gitignore` file is properly configured to **exclude all data files** from being committed to GitHub. This protects large datasets and keeps your repository lightweight.

---

## What Will NOT Be Uploaded (Protected)

### 📁 Data Files (~2.1+ GB)
```
✅ data/raw/*              (~1.0 GB)
   - en.openfoodfacts.org.products.tsv
   - USDA SR28 datasets
   - All other raw data files

✅ data/processed/*        (~17 MB)
   - *.parquet files
   - All processed datasets
```

### 🤖 Trained Models (~1.1 GB)
```
✅ models/*.pkl
✅ models/*.joblib
✅ models/*.h5
✅ models/*.pt
   - All trained model files
```

### 📊 Results (~13 MB)
```
✅ results/*
   - Visualizations
   - Performance metrics
   - Analysis outputs
```

### 📝 Internal Documentation (~9.3 MB)
```
✅ not_public/*
   - FINAL_PROJECT_REPORT.md
   - Internal verification reports
   - Development guides
```

### 💡 Other Protected Files
```
✅ __pycache__/
✅ .ipynb_checkpoints/
✅ .env, .venv
✅ .DS_Store, .vscode/
```

---

## What WILL Be Uploaded (Public)

### ✅ Source Code
- `src/` - All Python modules
- `*.py` - Training, prediction, optimization scripts
- `requirements.txt` - Dependencies

### ✅ Documentation
- `README.md` - Project documentation
- `GITHUB_UPLOAD_CHECKLIST.md`
- `GITHUB_PREPARATION_SUMMARY.md`
- `PROJECT_RENAME_SUMMARY.md`
- This file (`DATA_EXCLUSION_GUIDE.md`)

### ✅ Notebooks
- `notebooks/*.ipynb` - Jupyter notebooks (code only, no large outputs)

### ✅ Directory Structure
- `.gitkeep` files to preserve empty directories

---

## Verification Before Upload

Run these commands to verify what will be committed:

### Check .gitignore Status
```bash
cd /Users/sowmyasrinath/MSAI/CSML/final_project
cat .gitignore
```

### See What Git Will Track
```bash
git status --ignored
```

### List Files to Be Committed
```bash
git add .
git status
```

### Verify Data Files Are Ignored
```bash
# These should return "ignored" or "not tracked"
git check-ignore data/raw/*
git check-ignore data/processed/*
git check-ignore models/*.pkl
git check-ignore results/*
git check-ignore not_public/*
```

---

## If You Need to Share Data

Since data files won't be in GitHub, users will need instructions to obtain them:

### Option 1: External Storage (Recommended)
Upload to:
- Google Drive
- Zenodo (for research data)
- Figshare
- AWS S3 (if you have infrastructure)

### Option 2: Git LFS (Not Recommended for >1GB)
Git Large File Storage can handle large files, but has bandwidth limits.

### Option 3: Download Instructions (Current)
Your README.md already includes instructions for users to:
1. Download Open Food Facts from Kaggle
2. Download USDA SR28 from USDA website
3. Run preprocessing scripts to generate data

---

## Current .gitignore Configuration

```gitignore
# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Models
models/*.pkl
models/*.h5
models/*.pt
models/*.joblib
!models/.gitkeep

# Results
results/*
!results/.gitkeep

# Internal/private files
not_public/
```

**Key Pattern:**
- `data/raw/*` = Ignore everything in directory
- `!data/raw/.gitkeep` = EXCEPT the .gitkeep file (preserves directory structure)

---

## Safety Check: What Gets Uploaded

**Estimated repository size WITHOUT data:**
- Source code: ~500 KB
- Documentation: ~200 KB
- Notebooks: ~2 MB
- **Total: ~3-5 MB** (lightweight and fast to clone!)

**What you're AVOIDING uploading:**
- Data: ~2.1 GB
- Models: ~1.1 GB
- **Total saved: ~3+ GB** ✅

---

## Final Verification Script

Before pushing to GitHub, run this:

```bash
cd /Users/sowmyasrinath/MSAI/CSML/final_project

# Initialize git (if not done)
git init

# Add all files (respecting .gitignore)
git add .

# Check what will be committed
echo "=== FILES TO BE COMMITTED ==="
git status

echo ""
echo "=== VERIFYING DATA EXCLUSION ==="
echo "Checking if data files are ignored..."

if git status --ignored | grep -q "data/raw"; then
    echo "✅ data/raw/ is properly ignored"
else
    echo "❌ WARNING: data/raw/ might not be ignored!"
fi

if git status --ignored | grep -q "data/processed"; then
    echo "✅ data/processed/ is properly ignored"
else
    echo "❌ WARNING: data/processed/ might not be ignored!"
fi

if git status --ignored | grep -q "models"; then
    echo "✅ models/ is properly ignored"
else
    echo "❌ WARNING: models/ might not be ignored!"
fi

echo ""
echo "=== READY TO PUSH ==="
echo "If all checks pass, proceed with:"
echo "  git commit -m 'Initial commit: NutriSenseAI'"
echo "  git remote add origin <your-repo-url>"
echo "  git push -u origin main"
```

---

## Summary

✅ **You're all set!** Your data files are protected and will NOT be uploaded to GitHub.

Your repository will be:
- 🚀 Fast to clone (~3-5 MB vs ~3+ GB)
- 💰 Free (within GitHub limits)
- 🔒 Secure (no large data exposure)
- 📖 Professional (clean, well-documented codebase)

Users can still reproduce your work by downloading data from the original sources using your documented instructions.

---

**NutriSenseAI** - Making nutrition data-driven and accessible! 🥗🤖

