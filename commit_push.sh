#!/bin/bash
cd /home/pi/WF-AOI/wf-ai-dashboard
git add .
git commit -m "feat: add MQTT debug tool, ignore template files, clean up test scripts" > git_commit_output.txt 2>&1
git push origin main >> git_commit_output.txt 2>&1
echo "DONE" >> git_commit_output.txt
