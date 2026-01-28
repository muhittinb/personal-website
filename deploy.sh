#!/bin/bash

echo "🚀 Starting deployment..."

# 1. Back up Source Code to GitHub
git add .
msg="Site update $(date)"
if [ -n "$1" ]; then
    msg="$1"
fi
git commit -m "$msg"
git push origin master

# 2. Publish to the Web
quarto publish gh-pages

echo "✅ Done! Website is updating."
