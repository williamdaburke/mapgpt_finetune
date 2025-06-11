#!/bin/bash

# Check if minimum URL is provided as argument
if [ $# -lt 1 ]; then
    echo "Usage: $0 <URL> [PATH]"
    exit 1
fi

URL=$1
PATH_TO_APPEND=${2:-} # Optional path, defaults to empty if not provided

# Read credentials from ~/.ngrok_http
if [ ! -f ~/.ngrok_http ]; then
    echo "Error: ~/.ngrok_http not found. Make sure credentials are stored in the format 'username password'."
    exit 1
fi
read -r username password < ~/.ngrok_http

# Generate Basic authentication token
basic_auth_token=$(echo -n "$username:$password" | base64)

# Create a temporary Python script for mitmproxy
PY_SCRIPT=$(mktemp /tmp/inline_mitmproxy_XXXX.py)

# Generate the Python script with the provided path
cat > "$PY_SCRIPT" << EOF
from mitmproxy import http
def request(flow: http.HTTPFlow) -> None:
    # Append the provided path to all requests
    flow.request.path = "$PATH_TO_APPEND" + flow.request.path
EOF

# Start mitmproxy with the dynamically created Python script
mitmproxy --mode reverse:$URL --modify-headers ":~q:Authorization: Basic $basic_auth_token" --modify-headers ":~q:ngrok-skip-browser-warning: asdf" -s "$PY_SCRIPT" --listen-port 6969

# Clean up: Remove the temporary Python script after mitmproxy exits
rm "$PY_SCRIPT"
