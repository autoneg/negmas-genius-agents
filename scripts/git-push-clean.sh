#!/bin/bash
# Custom git push script that:
# 1. Removes [tool.uv.sources] section from pyproject.toml
# 2. Commits if needed (amend or new commit)
# 3. Pushes to remote
# 4. Restores [tool.uv.sources] section
# 5. Runs uv sync --all-extras --dev

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYPROJECT="$REPO_ROOT/pyproject.toml"

cd "$REPO_ROOT"

# Check if [tool.uv.sources] section exists
if grep -q '^\[tool\.uv\.sources\]' "$PYPROJECT"; then
    echo "Found [tool.uv.sources] section - removing for push..."
    
    # Save the sources section content
    SOURCES_SECTION=$(python3 -c "
import re
with open('$PYPROJECT', 'r') as f:
    content = f.read()
match = re.search(r'(\n\[tool\.uv\.sources\][^\[]*)', content)
if match:
    print(match.group(1))
")
    
    # Remove the sources section
    python3 -c "
import re
with open('$PYPROJECT', 'r') as f:
    content = f.read()
content = re.sub(r'\n\[tool\.uv\.sources\][^\[]*', '', content)
content = content.rstrip() + '\n'
with open('$PYPROJECT', 'w') as f:
    f.write(content)
"
    
    echo "Removed [tool.uv.sources] section"
    
    # Check if pyproject.toml has changes
    if ! git diff --quiet "$PYPROJECT"; then
        echo "Staging cleaned pyproject.toml..."
        git add "$PYPROJECT"
        
        # Check if there are other staged changes
        if git diff --cached --quiet; then
            echo "No other staged changes, amending last commit..."
            git commit --amend --no-edit
        else
            echo "Creating temporary commit for clean pyproject.toml..."
            git commit -m "chore: remove local dev sources for push"
        fi
    fi
    
    NEEDS_RESTORE=true
else
    echo "No [tool.uv.sources] section found - pushing as-is"
    NEEDS_RESTORE=false
fi

# Push to remote (pass through any arguments)
echo "Pushing to remote..."
git push "$@"
PUSH_STATUS=$?

# Restore the sources section if we removed it
if [ "$NEEDS_RESTORE" = true ]; then
    echo "Restoring [tool.uv.sources] section..."
    
    # Re-add the sources section
    echo "$SOURCES_SECTION" >> "$PYPROJECT"
    
    # Sync with uv
    echo "Running uv sync --all-extras --group dev..."
    uv sync --all-extras --group dev
    
    echo "Local environment restored!"
fi

exit $PUSH_STATUS
