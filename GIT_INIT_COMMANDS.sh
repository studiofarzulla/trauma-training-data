#!/bin/bash
# Trauma Models - GitHub Publication Commands
# Run these commands to publish the repository

echo "===== TRAUMA MODELS - GITHUB PUBLICATION ====="
echo ""
echo "Step 1: Initialize git repository"
cd /home/kawaiikali/Documents/Resurrexi/coding-with-buddy/trauma-models
git init
echo ""

echo "Step 2: Add all files (respecting .gitignore)"
git add .
echo ""

echo "Step 3: Review what will be committed"
git status
echo ""
echo "Files to be committed (excluding venv and session docs):"
git diff --cached --name-status | head -20
echo ""
read -p "Continue with commit? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Step 4: Create initial commit"
    git commit -m "Initial commit: Trauma as Training Data computational models

- 4 working experiments (extreme penalty, noisy signals, limited dataset, catastrophic forgetting)
- Publication-ready README with full documentation
- MIT License for open research
- Citation.cff for GitHub citation support
- Clean codebase with no cache files or sensitive data

Paper: 'Childhood Trauma as Training Data: A Machine Learning Framework'
Author: Murad Farzulla (2025)"
    
    echo ""
    echo "===== NEXT STEPS ====="
    echo "1. Create repository on GitHub: https://github.com/new"
    echo "   Name: trauma-models"
    echo "   Description: Computational models demonstrating how ML training dynamics mirror trauma formation mechanisms"
    echo ""
    echo "2. Connect to remote (replace YOUR_USERNAME):"
    echo "   git remote add origin https://github.com/YOUR_USERNAME/trauma-models.git"
    echo "   git branch -M main"
    echo "   git push -u origin main"
    echo ""
    echo "3. Update placeholders in files:"
    echo "   - README.md: Replace 'yourusername' with your GitHub username"
    echo "   - CITATION.cff: Add your ORCID and update GitHub URLs"
    echo ""
else
    echo "Commit cancelled. Review files with 'git status'"
fi
