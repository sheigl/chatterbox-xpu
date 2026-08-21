#!/bin/bash
# Ad-hoc launcher. The real deployment is the systemd user unit
# ~/.config/systemd/user/chatterbox.service -- keep --devices in sync there.
#
# usage: ./run.sh [device_pool]   e.g. ./run.sh xpu:2  or  ./run.sh xpu:0,xpu:1,xpu:2
#   [device_pool] : comma-separated XPU pool (default: $CHATTERBOX_DEVICES or xpu:2)
#   $PORT         : listen port (default 8045)
set -e

DEVICES="${1:-${CHATTERBOX_DEVICES:-xpu:0}}"
PORT="${PORT:-8045}"

./.venv/bin/python openai_server.py --host 0.0.0.0 --port "$PORT" --devices "$DEVICES"

