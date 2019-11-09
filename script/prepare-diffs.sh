#!/bin/bash
echo "Started preparing diffs..."

echo "Step 1/2: cloning repositories..."
python /workspace/src/clone_repository.py --repo_file /workspace/data/repositories_list.txt --output /tmp/clone_repositories
echo "Step 1/2: finished"

echo "Step 2/2: parsing diffs from clones repositories..."
python /workspace/src/download_diffs.py --repo_dir /tmp/clone_repositories --output /workspace/tmp/test2
echo "Step 2/2: finished"

echo "Parsed diffs are saved at /workspace/tmp/test2"