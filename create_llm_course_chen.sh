#!/bin/bash

# Check if gh is installed
if ! command -v gh &> /dev/null
then
    echo "GitHub CLI (gh) is not installed. Please install it first."
    exit 1
fi

# Check if logged in to GitHub
if ! gh auth status &> /dev/null
then
    echo "Please login to GitHub using 'gh auth login' before running this script."
    exit 1
fi

# Create main project directory
mkdir llm-course-chen && cd llm-course-chen

# Create src directory and its subdirectories
mkdir -p src/{nlp,llm,inference,prompt_engineering,quantization}

# Create notebooks directory
mkdir -p notebooks

# Define lesson names and details
declare -A lessons
lessons=(
    ["lesson01"]="Course Overview"
    ["lesson02"]="NLP Fundamentals"
    ["lesson03"]="Basic knowledge and architectural characteristics of LLM"
    ["lesson04"]="LLM Development Fundamentals"
    ["lesson05"]="Introduction and Setup of the Experimental Environment"
    ["lesson12"]="Model Inference and Function calling"
    ["lesson13"]="Prompt engineering - ChatGPT Prompt Engineering"
    ["lesson14"]="Model Quantization Techniques"
)

# Create notebooks for each lesson
for lesson in "${!lessons[@]}"
do
    touch "notebooks/${lesson}_${lessons[$lesson]// /_}.ipynb"
done

# Create additional directories
mkdir -p {data,docs,tests,config}

# Create root level files
touch {README.md,requirements.txt,Dockerfile,.gitignore}

# Create README content
cat << EOF > README.md
# LLM Course by Chen's Team

## Course Overview

This repository contains materials for the LLM course taught by Chen's team. The course covers fundamental concepts of Natural Language Processing (NLP), Large Language Models (LLMs), and their applications.

## Lesson Structure

1. Course Overview (0.2 hours)
2. NLP Fundamentals (1.0 hour)
3. Basic knowledge and architectural characteristics of LLM (0.5 hours)
4. LLM Development Fundamentals (1.0 hour)
5. Introduction and Setup of the Experimental Environment (0.2 hours)
12. Model Inference and Function calling (1.0 hour)
13. Prompt engineering - ChatGPT Prompt Engineering (1.0 hour)
14. Model Quantization Techniques (1.0 hour)

## Getting Started

[Provide instructions on how to use this repository and set up the environment]

## Contact

For any questions or clarifications, please contact [Your Contact Information].
EOF

echo "LLM Course project directory structure created successfully!"

# Initialize git repository
git init

# Add all files to git
git add .

# Commit changes
git commit -m "Initial project structure for LLM course"

# Create GitHub repository
gh repo create chenxingqiang/llm-course-chen --public --description "LLM Course by Chen's Team" --source=. --remote=origin --push

echo "GitHub repository created and initial commit pushed successfully!"
