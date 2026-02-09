# GitHub Repo Connection & Clone – Step-by-Step Guide

This guide walks you through connecting your machine to GitHub (with Credential Manager), cloning this repo, and using the **main** branch.

---

## Overview

1. **Install Git** (if needed) and configure **Credential Manager**
2. **Configure Git** (name & email)
3. **Clone the repo** into your workspace
4. **Work on the `main` branch**

---

## Step 1: Install Git & Credential Manager

- **Windows:** Download and install [Git for Windows](https://git-scm.com/download/win).
- During setup, **keep the default "Git Credential Manager"** so GitHub sign-in works over HTTPS.
- After installing, **close and reopen** PowerShell/terminal so `git` is in your PATH.

Check that Git is installed:

```powershell
git --version
```

---

## Step 2: Configure Git (Identity)

Git needs your name and email for commits. Run once per machine (use your GitHub name and email):

```powershell
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

---

## Step 3: Set Up GitHub Credentials (Credential Manager)

Git for Windows uses **Git Credential Manager**. The first time you clone or push to GitHub:

- A **browser window** will open.
- **Sign in to GitHub** and approve the app.
- Credentials are stored securely and reused.

To confirm the credential helper:

```powershell
git config --global credential.helper
```

You should see `manager` or `manager-core`. If not:

```powershell
git config --global credential.helper manager
```

**This project:**

- **Repo:** `https://github.com/Hellodevloops/jan-reconcilations-.git`
- **Branch:** `main`

---

## Step 4: Clone the Repo

Open **PowerShell** (or Command Prompt) in a folder where you want the project.

**Option A – Clone into current folder (e.g. `c:\ai-reconcilations`):**

```powershell
cd c:\ai-reconcilations
git clone https://github.com/Hellodevloops/jan-reconcilations-.git .
```

The `.` means “current directory”; all repo files will appear in that folder.

**Option B – Clone into a new folder:**

```powershell
cd c:\
git clone https://github.com/Hellodevloops/jan-reconcilations-.git ai-reconcilations
cd ai-reconcilations
```

If GitHub asks for login, use the browser window that opens.

---

## Step 5: Confirm Branch (main)

Check current branch:

```powershell
git branch
```

You should see `* main`. If not:

```powershell
git checkout main
```

---

## Step 6: Daily Workflow

- **Get latest:** `git pull origin main`
- **Save and push changes:**
  ```powershell
  git add .
  git commit -m "Your message"
  git push origin main
  ```

---

## Quick Reference

| Item         | Value                                                                 |
|-------------|-----------------------------------------------------------------------|
| Repo URL    | `https://github.com/Hellodevloops/jan-reconcilations-.git`           |
| Branch      | `main`                                                                |
| Credentials | Git Credential Manager (browser sign-in first time)                   |

---

## Troubleshooting

- **"git is not recognized"**  
  Install Git for Windows and **reopen** your terminal.

- **"Authentication failed"**  
  Run `git config --global credential.helper manager` and try again; sign in via the browser when prompted.

- **"Repository not found"**  
  Check the URL and that your GitHub account has access to `Hellodevloops/jan-reconcilations-`.
