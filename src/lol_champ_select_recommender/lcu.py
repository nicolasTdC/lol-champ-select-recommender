from __future__ import annotations

import base64
import json
import os
import platform
import shutil
import ssl
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class LcuError(RuntimeError):
    """Raised when the local League Client API cannot be reached."""


@dataclass(frozen=True)
class LcuConnection:
    port: int
    password: str
    protocol: str = "https"
    host: str = "127.0.0.1"
    transport: str = "python"

    @property
    def base_url(self) -> str:
        return f"{self.protocol}://{self.host}:{self.port}"

    def with_host(self, host: str) -> LcuConnection:
        return LcuConnection(
            port=self.port,
            password=self.password,
            protocol=self.protocol,
            host=host,
            transport=self.transport,
        )

    def with_windows_curl(self) -> LcuConnection:
        return LcuConnection(
            port=self.port,
            password=self.password,
            protocol=self.protocol,
            host="127.0.0.1",
            transport="windows-curl",
        )

    def request_json(self, path: str, timeout: float = 2.0) -> Any:
        if self.transport == "windows-curl":
            return self._request_json_with_windows_curl(path, timeout)

        url = f"{self.base_url}{path}"
        token = base64.b64encode(f"riot:{self.password}".encode("utf-8")).decode("ascii")
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Basic {token}",
                "Accept": "application/json",
                "User-Agent": "lol-champ-select-recommender/0.1",
            },
        )
        context = ssl._create_unverified_context()

        try:
            with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LcuError(f"LCU returned HTTP {exc.code} for {path}: {detail}") from exc
        except OSError as exc:
            raise LcuError(f"Could not reach League Client API at {url}: {exc}") from exc

        if not body:
            return None

        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise LcuError(f"LCU returned non-JSON data for {path}") from exc

    def _request_json_with_windows_curl(self, path: str, timeout: float = 2.0) -> Any:
        curl_path = windows_curl_path()
        if not curl_path:
            raise LcuError("Windows curl.exe was not found from WSL.")

        url = f"{self.base_url}{path}"
        token = base64.b64encode(f"riot:{self.password}".encode("utf-8")).decode("ascii")
        marker = "\n__LCU_HTTP_STATUS__:"
        command = [
            curl_path,
            "-k",
            "-sS",
            "--connect-timeout",
            str(max(1, int(timeout))),
            "--max-time",
            str(max(2, int(timeout + 2))),
            "-H",
            f"Authorization: Basic {token}",
            "-H",
            "Accept: application/json",
            "-H",
            "User-Agent: lol-champ-select-recommender/0.1",
            "-w",
            f"{marker}%{{http_code}}",
            url,
        ]

        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout + 3,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LcuError(f"Could not run Windows curl.exe for {url}: {exc}") from exc

        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise LcuError(f"Could not reach League Client API through Windows curl at {url}: {detail}")

        if marker not in result.stdout:
            raise LcuError(f"Windows curl.exe returned an unexpected response for {path}")

        body, status_raw = result.stdout.rsplit(marker, 1)
        try:
            status = int(status_raw.strip())
        except ValueError as exc:
            raise LcuError(f"Windows curl.exe returned an invalid HTTP status for {path}: {status_raw}") from exc

        if status >= 400:
            raise LcuError(f"LCU returned HTTP {status} for {path}: {body.strip()}")

        if not body:
            return None

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise LcuError(f"LCU returned non-JSON data for {path}") from exc

    def gameflow_phase(self) -> str:
        phase = self.request_json("/lol-gameflow/v1/gameflow-phase")
        return str(phase or "None")

    def champ_select_session(self) -> dict[str, Any] | None:
        try:
            session = self.request_json("/lol-champ-select/v1/session")
        except LcuError as exc:
            message = str(exc)
            if "HTTP 404" in message or "HTTP 400" in message:
                return None
            raise
        return session if isinstance(session, dict) else None


def find_lockfile(explicit_path: str | None = None) -> Path | None:
    candidates: list[Path] = []

    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    for env_name in ("LOL_LOCKFILE", "LCU_LOCKFILE"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value).expanduser())

    candidates.extend(default_lockfile_paths())

    for path in candidates:
        if path.is_file():
            return path

    return None


def default_lockfile_paths() -> list[Path]:
    system = platform.system().lower()
    home = Path.home()

    paths = [
        Path.cwd() / "lockfile",
        home / "Riot Games" / "League of Legends" / "lockfile",
        Path("/mnt/c/Riot Games/League of Legends/lockfile"),
        Path("/mnt/c/Program Files/Riot Games/League of Legends/lockfile"),
        Path("/mnt/c/Program Files (x86)/Riot Games/League of Legends/lockfile"),
        home / ".wine" / "drive_c" / "Riot Games" / "League of Legends" / "lockfile",
        home / "Games" / "league-of-legends" / "drive_c" / "Riot Games" / "League of Legends" / "lockfile",
    ]

    if system == "windows":
        paths.extend(
            [
                Path("C:/Riot Games/League of Legends/lockfile"),
                Path("C:/Program Files/Riot Games/League of Legends/lockfile"),
                Path("C:/Program Files (x86)/Riot Games/League of Legends/lockfile"),
            ]
        )
    elif system == "darwin":
        paths.append(Path("/Applications/League of Legends.app/Contents/LoL/lockfile"))

    return paths


def connection_from_lockfile(path: Path, host: str = "127.0.0.1") -> LcuConnection:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise LcuError(f"Could not read lockfile at {path}: {exc}") from exc

    parts = raw.split(":")
    if len(parts) != 5:
        raise LcuError(f"Unexpected lockfile format at {path}")

    _, _, port_raw, password, protocol = parts

    try:
        port = int(port_raw)
    except ValueError as exc:
        raise LcuError(f"Unexpected LCU port in lockfile at {path}: {port_raw}") from exc

    return LcuConnection(port=port, password=password, protocol=protocol, host=host)


def connection_from_process_args(host: str = "127.0.0.1") -> LcuConnection | None:
    if platform.system().lower() == "windows":
        return None

    try:
        result = subprocess.run(
            ["ps", "-axo", "command"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    for line in result.stdout.splitlines():
        if "LeagueClientUx" not in line:
            continue

        port = _extract_process_arg(line, "--app-port=")
        password = _extract_process_arg(line, "--remoting-auth-token=")
        if port and password:
            try:
                return LcuConnection(port=int(port), password=password, protocol="https", host=host)
            except ValueError:
                return None

    return None


def connect(explicit_lockfile: str | None = None, host: str | None = None) -> tuple[LcuConnection, Path | None]:
    hosts = connection_hosts(host)
    lockfile = find_lockfile(explicit_lockfile)
    if lockfile:
        connection = connection_from_lockfile(lockfile, hosts[0])
        return _first_reachable_connection(connection, hosts), lockfile

    process_connection = connection_from_process_args(hosts[0])
    if process_connection:
        return _first_reachable_connection(process_connection, hosts), None

    raise LcuError(
        "League lockfile was not found. Start the League client or pass --lockfile /path/to/lockfile."
    )


def connection_hosts(explicit_host: str | None = None) -> list[str]:
    configured_host = explicit_host or os.environ.get("LCU_HOST")
    if configured_host:
        return [configured_host]

    hosts = ["127.0.0.1"]
    if is_wsl():
        hosts.extend(wsl_windows_host_candidates())

    return _dedupe(hosts)


def is_wsl() -> bool:
    release = platform.release().lower()
    if "microsoft" in release or "wsl" in release:
        return True

    try:
        version = Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False

    return "microsoft" in version or "wsl" in version


def wsl_windows_host_candidates() -> list[str]:
    candidates: list[str] = []

    try:
        for line in Path("/etc/resolv.conf").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "nameserver":
                candidates.append(parts[1])
    except OSError:
        pass

    try:
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if "via" in parts:
                candidates.append(parts[parts.index("via") + 1])
    except (OSError, subprocess.TimeoutExpired):
        pass

    return _dedupe(candidates)


def _first_reachable_connection(connection: LcuConnection, hosts: list[str]) -> LcuConnection:
    errors: list[str] = []

    for host in hosts:
        candidate = connection.with_host(host)
        try:
            candidate.gameflow_phase()
            return candidate
        except LcuError as exc:
            errors.append(f"{host}: {exc}")

    if is_wsl() and windows_curl_path():
        candidate = connection.with_windows_curl()
        try:
            candidate.gameflow_phase()
            return candidate
        except LcuError as exc:
            errors.append(f"windows-curl: {exc}")

    if len(hosts) > 1:
        tried = ", ".join(hosts)
        raise LcuError(
            "Could not reach League Client API using any candidate host. "
            f"Tried: {tried}. Last error: {errors[-1]}"
        )

    raise LcuError(errors[-1] if errors else "Could not reach League Client API.")


def windows_curl_path() -> str | None:
    path = shutil.which("curl.exe")
    if path:
        return path

    for candidate in (
        Path("/mnt/c/Windows/System32/curl.exe"),
        Path("/mnt/c/WINDOWS/system32/curl.exe"),
    ):
        if candidate.is_file():
            return str(candidate)

    return None


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = value.strip()
        if not value or value in seen:
            continue
        result.append(value)
        seen.add(value)
    return result


def _extract_process_arg(command: str, prefix: str) -> str | None:
    for part in command.split():
        if part.startswith(prefix):
            return part[len(prefix) :].strip('"')
    return None
