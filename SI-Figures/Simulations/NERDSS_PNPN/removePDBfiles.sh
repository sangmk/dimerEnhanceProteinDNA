find . -type d -name "PDB" -print0 | \
xargs -0 -P 8 -I {} bash -c 'echo "Clearing contents of: {}"; rm -rf "{}"/* "{}"/.[!.]* "{}"/..?*'
