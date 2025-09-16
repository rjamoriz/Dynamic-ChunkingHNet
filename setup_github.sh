#!/bin/bash

# 🚀 H-Net Dynamic Chunking - GitHub Setup Script
# This script will help you create and push your repository to GitHub

echo "🚀 H-Net Dynamic Chunking - GitHub Repository Setup"
echo "=================================================="

# Step 1: Authenticate with GitHub
echo ""
echo "📋 Step 1: GitHub Authentication"
echo "You need to authenticate with GitHub CLI first."
echo "Run: gh auth login"
echo "Choose: GitHub.com → HTTPS → Login via web browser"
echo ""
read -p "Press Enter after you've completed GitHub authentication..."

# Step 2: Create the repository
echo ""
echo "📋 Step 2: Creating GitHub Repository"
echo "Creating repository: Dynamic-ChunkingHNet"

gh repo create Dynamic-ChunkingHNet \
  --public \
  --description "🚀 H-Net Dynamic Chunking Implementation - Advanced text segmentation for RAG systems with interactive Apache ECharts visualizations" \
  --clone=false

if [ $? -eq 0 ]; then
    echo "✅ Repository created successfully!"
else
    echo "❌ Failed to create repository. Please check your authentication."
    exit 1
fi

# Step 3: Add remote origin
echo ""
echo "📋 Step 3: Adding remote origin"
git remote add origin https://github.com/$(gh api user --jq .login)/Dynamic-ChunkingHNet.git

# Step 4: Push to GitHub
echo ""
echo "📋 Step 4: Pushing to GitHub"
git branch -M main
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! Your H-Net Dynamic Chunking repository is now on GitHub!"
    echo "🔗 Repository URL: https://github.com/$(gh api user --jq .login)/Dynamic-ChunkingHNet"
    echo ""
    echo "📊 What's included:"
    echo "   ✅ Complete H-Net implementation notebook"
    echo "   ✅ Interactive Apache ECharts visualizations"
    echo "   ✅ Research paper analysis"
    echo "   ✅ Professional documentation"
    echo "   ✅ Requirements and setup files"
    echo ""
    echo "🚀 Your repository is ready for collaboration and sharing!"
else
    echo "❌ Failed to push to GitHub. Please check the remote URL and try again."
fi
