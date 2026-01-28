#!/bin/bash

# 1. Move to the folder where this script is located
cd "$(dirname "$0")"

echo "🚀 Starting deployment from $(pwd)..."

# 2. Back up Source Code to GitHub
git add .
msg="Site update $(date)"
if [ -n "$1" ]; then
    msg="$1"
fi
git commit -m "$msg"
git push origin master

# 3. Publish to the Web
quarto publish gh-pages

echo "✅ Done! Website is updating."