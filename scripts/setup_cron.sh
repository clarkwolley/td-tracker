#!/usr/bin/env bash
#
# Install (or remove) the TD Tracker weekly cron job.
#
# Runs every Tuesday at 8:00 AM — after Monday Night Football,
# when all weekly results are final and nflverse data is updated.
#
# Usage:
#   ./scripts/setup_cron.sh install
#   ./scripts/setup_cron.sh remove
#   ./scripts/setup_cron.sh status

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
PYTHON="${PROJECT_DIR}/venv/bin/python"
MARKER="# td-tracker-weekly"

CRON_LINE="0 8 * * 2 cd ${PROJECT_DIR} && ${PYTHON} -m src.automation >> ${LOG_DIR}/pipeline.log 2>&1 ${MARKER}"

install_cron() {
    mkdir -p "${LOG_DIR}"

    if crontab -l 2>/dev/null | grep -q "${MARKER}"; then
        echo "⚠️  Cron job already installed. Use 'remove' first to reinstall."
        status_cron
        return
    fi

    (crontab -l 2>/dev/null; echo "${CRON_LINE}") | crontab -
    echo "✅ Cron job installed!"
    echo "   Schedule: Every Tuesday at 8:00 AM"
    echo "   Logs:     ${LOG_DIR}/pipeline.log"
    echo ""
    echo "   To test now:  ${PYTHON} -m src.automation --no-notify"
}

remove_cron() {
    if ! crontab -l 2>/dev/null | grep -q "${MARKER}"; then
        echo "ℹ️  No td-tracker cron job found."
        return
    fi

    crontab -l 2>/dev/null | grep -v "${MARKER}" | crontab -
    echo "🗑️  Cron job removed."
}

status_cron() {
    echo ""
    echo "Current td-tracker cron entries:"
    if crontab -l 2>/dev/null | grep "${MARKER}"; then
        echo "   ✅ Active"
    else
        echo "   ❌ Not installed"
    fi

    if [ -f "${LOG_DIR}/pipeline.log" ]; then
        echo ""
        echo "Last 5 log lines:"
        tail -5 "${LOG_DIR}/pipeline.log" | sed 's/^/   /'
    fi
}

case "${1:-status}" in
    install) install_cron ;;
    remove)  remove_cron ;;
    status)  status_cron ;;
    *)
        echo "Usage: $0 {install|remove|status}"
        exit 1
        ;;
esac
