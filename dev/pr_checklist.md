# 🔍 PR Checklist - Pre-Merge Health Check

Use this checklist before finalizing your branch and creating a pull request.

## ✅ Quick Health Check

### 1. **Run Full Health Check**
```bash
python scripts/check_health.py
```
This will show you:
- Lines of code
- Type violations (ty check)
- Format issues (ruff)
- Lint issues (ruff)

**Target:** Aim for 🟢 EXCELLENT or 🟡 GOOD health status

### 2. **Fix Critical Issues**
Based on health check results:

**Type Issues (if > 10 violations):**
```bash
# Check specific files with most violations
uvx ty check --output-format concise | head -20
```

**Format Issues (if any):**
```bash
# Auto-fix formatting
ruff format .
```

**Lint Issues (if > 100):**
```bash
# Check and fix auto-fixable issues
ruff check --fix .
```

## 🧪 Testing

### 3. **Run Fast Unit Tests**
```bash
# Quick test run (should complete in ~3 seconds)
python dev/update_readme_metrics.py --fast
```

### 4. **Run Full Test Suite** (if needed)
```bash
# Complete test run (takes longer)
python dev/update_readme_metrics.py
```

### 5. **Check for Slow Tests**
```bash
# Generate test timing report
python dev/run_test_durations.py
```

## 🧹 Code Quality

### 6. **Review Changes**
```bash
# See what files you've modified
git status

# Review your changes
git diff main --name-only
```

### 7. **Clean Up Temporary Files**
Remove any experimental/temporary files:
```bash
# Check for junk files in root
ls -la *.py | grep -v conftest

# Remove if found
rm -f temp_*.py debug_*.py test_*.py
```

## 📦 Version & Release (if applicable)

### 8. **Version Management**
If your changes warrant a version bump:
```bash
# Dev version increment
python scripts/release.py --dry-run

# Minor version increment  
python scripts/release.py --minor --dry-run
```

## 🔀 Pre-Merge

### 9. **Merge Main into Your Branch**
```bash
# Update your branch with latest main
git checkout your-branch
git merge main
```

### 10. **Final Status Check**
```bash
git status
git log --oneline -5
```

## 📋 Checklist Summary

Before creating your PR, ensure:

- [ ] **Health check passes** (🟢 or 🟡 status)
- [ ] **Fast tests pass** (< 10 seconds)
- [ ] **No obvious format issues** (ruff format clean)
- [ ] **Major lint issues addressed** (< 100 violations)
- [ ] **No junk files** in root directory
- [ ] **Branch merged with main** (no conflicts)
- [ ] **Clean git status** (no untracked important files)
- [ ] **Meaningful commit messages**

## 🚀 Ready to Merge!

Your branch is now ready for:
1. **Pull Request creation**
2. **Code review**
3. **Merge to main**

---

## 🆘 Common Issues

**Health check fails?**
- Focus on files with most violations
- Use `ruff format .` for quick formatting fixes
- Address type issues in high-violation files

**Tests are slow?**
- Use `--fast` flag for quick feedback
- Check `test_durations.txt` for slow tests
- Consider marking slow tests with `@pytest.mark.slow`

**Merge conflicts?**
- Resolve conflicts favoring your branch changes
- Re-run health check after resolving conflicts

**Need help?**
- Check `README.md` for detailed command usage
- Run commands with `--help` flag for options
