#!/usr/bin/env bash
# Install sdr-scanner as a systemd service.
# Run with: sudo bash install-service.sh
set -euo pipefail

SERVICE_NAME="sdr-scanner"
UNIT_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Resolve the user who actually owns the project ──────────────────────────
# When run with sudo, SUDO_USER is the original user; fall back to $USER.
RUN_AS="${SUDO_USER:-$USER}"
if [[ "$RUN_AS" == "root" ]]; then
    echo "WARNING: running the service as root is not recommended."
    echo "Re-run as a normal user:  sudo bash install-service.sh"
fi

# ── Find Python ─────────────────────────────────────────────────────────────
if   [[ -x "$SCRIPT_DIR/venv/bin/python" ]];  then PYTHON="$SCRIPT_DIR/venv/bin/python"
elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then PYTHON="$SCRIPT_DIR/.venv/bin/python"
elif command -v python3 &>/dev/null;           then PYTHON="$(command -v python3)"
else
    echo "ERROR: Python 3 not found. Install it or create a venv at $SCRIPT_DIR/venv." >&2
    exit 1
fi

echo "==> Service name : $SERVICE_NAME"
echo "==> Working dir  : $SCRIPT_DIR"
echo "==> Python       : $PYTHON"
echo "==> Run as user  : $RUN_AS"
echo ""

# ── RTL-SDR USB permissions ─────────────────────────────────────────────────
# librtlsdr needs direct USB access.  The standard approach is to put the user
# in the 'plugdev' group and rely on the udev rules installed by the rtl-sdr
# package (/etc/udev/rules.d/rtl-sdr.rules).  If those rules are missing we
# install a minimal one here.

UDEV_RULES="/etc/udev/rules.d/rtl-sdr.rules"
if [[ ! -f "$UDEV_RULES" ]]; then
    echo "==> Installing RTL-SDR udev rules → $UDEV_RULES"
    cat > "$UDEV_RULES" <<'UDEV'
# RTL-SDR USB devices — grant plugdev group read/write access
SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2832", GROUP="plugdev", MODE="0664"
SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", GROUP="plugdev", MODE="0664"
UDEV
    udevadm control --reload-rules
    udevadm trigger
    echo "    Udev rules installed."
fi

# Ensure the user is in plugdev
if ! id -nG "$RUN_AS" | grep -qw plugdev; then
    echo "==> Adding $RUN_AS to group 'plugdev'"
    usermod -aG plugdev "$RUN_AS"
    echo "    NOTE: the user must log out and back in (or reboot) for this"
    echo "    group membership to take effect before the device works."
fi

# Blacklist kernel DVB driver that conflicts with librtlsdr
BLACKLIST="/etc/modprobe.d/blacklist-rtl.conf"
if [[ ! -f "$BLACKLIST" ]]; then
    echo "==> Blacklisting dvb_usb_rtl28xxu kernel module → $BLACKLIST"
    echo "blacklist dvb_usb_rtl28xxu" > "$BLACKLIST"
    echo "blacklist rtl2832"           >> "$BLACKLIST"
    echo "blacklist rtl2830"           >> "$BLACKLIST"
fi

# ── Write the systemd unit ───────────────────────────────────────────────────
echo "==> Writing $UNIT_FILE"
cat > "$UNIT_FILE" <<UNIT
[Unit]
Description=SDR Scanner
Documentation=https://github.com/mostlychris/rtl-airband-scanner
After=network.target
Wants=network.target

[Service]
Type=simple
User=${RUN_AS}
WorkingDirectory=${SCRIPT_DIR}
ExecStart=${PYTHON} ${SCRIPT_DIR}/app.py
Restart=on-failure
RestartSec=5
# Give the USB subsystem time to settle after boot
TimeoutStartSec=30
StandardOutput=journal
StandardError=journal
# Prevent runaway restarts
StartLimitIntervalSec=120
StartLimitBurst=5

[Install]
WantedBy=multi-user.target
UNIT

# ── Enable and start ─────────────────────────────────────────────────────────
echo "==> Reloading systemd"
systemctl daemon-reload

echo "==> Enabling $SERVICE_NAME (auto-start on boot)"
systemctl enable "$SERVICE_NAME"

echo "==> Starting $SERVICE_NAME"
systemctl start "$SERVICE_NAME"

echo ""
echo "Done.  Useful commands:"
echo "  Status  : sudo systemctl status $SERVICE_NAME"
echo "  Logs    : sudo journalctl -u $SERVICE_NAME -f"
echo "  Restart : sudo systemctl restart $SERVICE_NAME"
echo "  Stop    : sudo systemctl stop $SERVICE_NAME"
echo "  Disable : sudo systemctl disable $SERVICE_NAME"
echo "  Remove  : sudo systemctl disable $SERVICE_NAME && sudo rm $UNIT_FILE"
echo ""
echo "The web UI will be available at  http://<device-ip>:8080"
echo "To change the port, edit $UNIT_FILE and add --listen-port N to ExecStart."
