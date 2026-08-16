"""Launch a self-hosted W&B Server and open your browser to inspect locally logged offline runs.

What it does:
    * **Autodetects a free port** (or respects `--port`) so you never collide with running services.
    * Uses the `wandb server start` command (v0.70+) with the `--daemon/--no-daemon` switch.
    * Wraps environment overrides in a **context-manager** that always restores the previous state.
    * Can launch via the **Python Docker SDK** (`--backend=sdk`) to avoid shell subprocesses.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Protocol
import webbrowser

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Generator
    from io import BufferedWriter
    from logging import Logger
    from types import ModuleType

    from docker import DockerClient
    from docker.models.containers import Container


# Optional: Python Docker SDK for --backend=sdk
has_docker: bool = False
docker_module: ModuleType | None = None
logger: Logger = logging.getLogger(__name__)
try:
    import docker
    from docker.errors import NotFound

    has_docker = True
    docker_module = docker
except ImportError:
    docker_module = None


class Closable(Protocol):
    """Minimal protocol for closeable streams."""

    def close(self) -> None:
        """Close the stream resource."""
        ...


class ServerProcess(Protocol):
    """Protocol for subprocess-based server backends."""

    def terminate(self) -> None:
        """Terminate the server process."""
        ...

    def kill(self) -> None:
        """Force kill the server process."""
        ...

    def poll(self) -> int | None:
        """Check if process is still running."""
        ...

    def wait(self, timeout: float | None = None) -> int:
        """Wait for process to terminate."""
        ...

    @property
    def stdout(self) -> Closable | None:
        """Standard output stream."""
        ...

    @property
    def stderr(self) -> Closable | None:
        """Standard error stream."""
        ...


class ServerContainer(Protocol):
    """Protocol for container-based server backends."""

    def remove(self, *, v: bool = False, link: bool = False, force: bool = False) -> None:
        """Remove the container.

        Args:
            v: Remove the volumes associated with the container.
            link: Remove the specified link and not the underlying container.
            force: Force the removal of a running container (uses SIGKILL).
        """
        ...

    def reload(self) -> None:
        """Reload the container information from the server."""
        ...

    @property
    def status(self) -> str:
        """Current status of the container."""
        ...


def docker_daemon_running() -> bool:
    """Check if the Docker daemon is running.

    Returns:
        bool: True if Docker daemon is running and accessible, False otherwise.
    """
    docker_executable = shutil.which("docker")
    if docker_executable is None:
        return False

    try:
        subprocess.run([docker_executable, "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
        return False
    else:
        return True


@contextlib.contextmanager
def temporary_env(**updates: str) -> Generator[None, None, None]:
    """Temporarily update environment variables and restore afterwards.

    Args:
        **updates: Environment variable names and values to update.

    Yields:
        None: Context manager yields nothing.
    """
    original = {k: os.environ.get(k) for k in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for k, v in original.items():
            if v is None:
                os.environ.pop(k, None)

            else:
                os.environ[k] = v


def is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a TCP connection can be made to the specified host and port.

    Args:
        port: The port number to check.
        host: The hostname or IP address to check. Defaults to "127.0.0.1".

    Returns:
        bool: True if a TCP connection can be established, False otherwise.
    """
    with socket.socket() as s:
        s.settimeout(0.4)
        try:
            s.connect((host, port))
        except OSError:
            return False
        else:
            return True


def find_free_port(preferred: int | None = None) -> int:
    """Find a free port for binding.

    Args:
        preferred: The preferred port number. If None or the port is already
            in use, an arbitrary free port will be returned.

    Returns:
        int: A free port number.
    """
    if preferred is not None and not is_port_open(preferred):
        return preferred

    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the W&B offline server.

    Args:
        argv: List of command line arguments. If None, uses sys.argv.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    p = argparse.ArgumentParser(description="Serve local W&B offline runs in your browser")
    p.add_argument(
        "-r",
        "--wandb-root",
        type=Path,
        default=Path.cwd(),
        help="Directory containing offline run folders (default: pwd)",
    )
    p.add_argument("-p", "--port", type=int, help="HTTP port to bind (default: auto-select)")
    p.add_argument("--no-browser", action="store_true", help="Do not open the default browser after startup")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase log verbosity. Use -vv for DEBUG")
    p.add_argument(
        "--backend",
        choices=("cli", "sdk"),
        default="cli",
        help=(
            "How to launch the server: `cli` (default) invokes the `wandb` CLI. "
            "`sdk` uses the Python Docker SDK to start the container without a shell subprocess."
        ),
    )
    return p.parse_args(argv)


def start_server(_root: Path, port: int) -> subprocess.Popen[bytes]:
    """Launch the W&B server in daemon mode and return the process handle.

    Args:
        _root: The root directory containing offline run folders.
        port: The port number to bind the server to.

    Returns:
        subprocess.Popen[bytes]: The subprocess handle for the running server.
    """
    cmd: list[str] = ["wandb", "server", "start", "--port", str(port), "--daemon"]
    logger.info(f"Executing: {' '.join(cmd)}")
    log_file: Path = Path.home() / ".wandb_server.log"
    log_handle: BufferedWriter = log_file.open("wb")
    logger.debug(f"Streaming server logs to {log_file}")
    return subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT)


# Python Docker SDK backend
def start_server_sdk(root: Path, port: int) -> ServerContainer:
    """Launch the W&B server via the Docker Python SDK.

    Args:
        root: The root directory containing offline run folders.
        port: The port number to bind the server to.

    Returns:
        ServerContainer: The Docker container handle for the running server.

    Raises:
        SystemExit: If Docker SDK is not available or Docker daemon is not accessible.
    """
    if not has_docker or docker_module is None:
        logger.error("Docker SDK not installed. Install it with `pip install docker` or use `--backend=cli`.")
        raise SystemExit(1)

    # Try to connect to Docker daemon with proper socket path for macOS Docker Desktop
    client: DockerClient | None = None
    docker_hosts: list[str | None] = [
        None,  # Default connection (will use DOCKER_HOST if set)
        "unix:///var/run/docker.sock",  # Standard Linux/Unix socket
        f"unix://{Path.home()}/Library/Containers/com.docker.docker/Data/docker-cli.sock",  # macOS Docker Desktop
    ]

    for host in docker_hosts:
        try:
            candidate: DockerClient = (
                docker_module.from_env() if host is None else docker_module.DockerClient(base_url=host)
            )
            candidate.ping()  # Test connection
            client = candidate
            logger.debug(f"Connected to the Docker daemon at {host or 'default'}")
            break

        except Exception as exc:
            logger.debug(f"Failed to connect to Docker daemon at {host or 'default'}: {exc}")
            client = None
            continue

    if client is None:
        logger.error("Cannot connect to the Docker daemon. Confirm that Docker is running and accessible.")
        raise SystemExit(1)

    image = "wandb/local"
    logger.info(f"Pulling image {image} (if needed)...")
    client.images.pull(image)

    container_name = f"wandb-local-{port}"
    # Clean up any previous container with same name
    try:
        old: Container = client.containers.get(container_name)
        logger.debug(f"Removing existing container {container_name}")
        old.remove(force=True)

    except NotFound:
        pass

    logger.info("Starting W&B Local container ...")
    return client.containers.run(
        image,
        detach=True,
        name=container_name,
        ports={"8080/tcp": port},
        volumes={str(root.absolute()): {"bind": "/vol", "mode": "rw"}},
        environment={"WANDB_MODE": "offline"},
    )


def wait_until_ready(port: int, timeout: float = 90.0, check_interval: float = 0.3) -> bool:
    """Block until the server answers on the specified port or timeout passes.

    Args:
        port: The port number to check for server readiness.
        timeout: Maximum time to wait in seconds. Defaults to 90.0.
        check_interval: How often to check in seconds. Defaults to 0.3.

    Returns:
        bool: True if server becomes ready within timeout, False otherwise.
    """
    logger.info(f"Waiting for W&B server to expose port {port} ...")
    start: float = time.perf_counter()

    while time.perf_counter() - start < timeout:
        if is_port_open(port):
            return True

        time.sleep(check_interval)

    return False


def _server_is_healthy(
    server_process: ServerProcess | None, server_container: ServerContainer | None, port: int
) -> bool:
    if server_process is not None:
        if server_process.poll() is not None:
            logger.error("W&B server process has died unexpectedly")
            return False

    elif server_container is not None:
        try:
            server_container.reload()  # Refresh container state
            if server_container.status != "running":
                logger.error(f"W&B container is no longer running (status: {server_container.status})")
                return False

        except Exception:
            logger.exception("Failed to check container status")
            return False

    if not is_port_open(port):
        logger.error(f"W&B server is no longer accessible on port {port}")
        return False

    return True


def monitor_server(server_process: ServerProcess | None, server_container: ServerContainer | None, port: int) -> None:
    """Monitor server and container health, cleaning up if they fail.

    Args:
        server_process: The subprocess handle for CLI backend, if any.
        server_container: The Docker container handle for SDK backend, if any.
        port: The port the server should be running on.
    """
    while True:
        try:
            if not _server_is_healthy(server_process, server_container, port):
                break

            time.sleep(5)  # Check every 5 seconds

        except KeyboardInterrupt:
            break


def cleanup_resources(server_process: ServerProcess | None, server_container: ServerContainer | None) -> None:
    """Stop the server process or remove its container.

    Args:
        server_process: The subprocess handle for CLI backend, if any.
        server_container: The Docker container handle for SDK backend, if any.
    """
    if server_process is not None:
        logger.info("Stopping W&B server process...")
        if server_process.poll() is None:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Server process exceeded the shutdown timeout, killing it")
                server_process.kill()
                server_process.wait()

        # Close file handles
        if server_process.stdout is not None:
            server_process.stdout.close()
        if server_process.stderr is not None:
            server_process.stderr.close()

    elif server_container is not None:
        logger.info("Stopping W&B container...")
        try:
            server_container.remove(force=True)

        except Exception as exc:
            logger.warning(f"Failed to remove container: {exc}")


def _run_ready_server(
    args: Namespace, port: int, server_process: subprocess.Popen[bytes] | None, server_container: ServerContainer | None
) -> None:
    if not wait_until_ready(port):
        logger.error("Server failed to start in time.")
        cleanup_resources(server_process, server_container)
        sys.exit(1)

    url: str = f"http://localhost:{port}"
    logger.info(f"Server ready: {url}")

    if not args.no_browser:
        webbrowser.open(url)

    # Monitor server and keep running until user interruption
    monitor_server(server_process, server_container, port)


def main(argv: list[str] | None = None) -> None:
    """Run the W&B offline server with the specified command line arguments.

    Args:
        argv: List of command line arguments. If None, uses sys.argv.

    """
    args: Namespace = parse_args(argv)

    # Safety check: --backend=sdk requires docker Python module
    if args.backend == "sdk" and not has_docker:
        print(
            "Selected --backend=sdk but the `docker` Python package is not installed. Falling back to --backend=cli.",
            file=sys.stderr,
        )
        args.backend = "cli"

    logging.basicConfig(
        level=logging.WARNING - (10 * min(args.verbose, 2)),
        format="{asctime} │ {levelname:8} │ {message}",
        datefmt="%H:%M:%S",
        style="{",
    )

    if not shutil.which("wandb"):
        logger.error("`wandb` CLI not found - please `pip install wandb`.")
        sys.exit(1)

    if not docker_daemon_running():
        logger.error(
            "Docker is required for `wandb server start`, but it is either not installed or the daemon isn't running. "
            "Install Docker Desktop (macOS/Windows) or docker-ce (Linux), then confirm that `docker info` succeeds."
        )
        sys.exit(1)

    if not args.wandb_root.exists():
        logger.error(f"Root directory {args.wandb_root} does not exist.")
        sys.exit(1)

    port: int = find_free_port(args.port)

    with temporary_env(WANDB_MODE="offline", WANDB_DIR=str(args.wandb_root)):
        server_process: subprocess.Popen[bytes] | None = None
        server_container: ServerContainer | None = None

        if args.backend == "sdk":
            try:
                server_container = start_server_sdk(args.wandb_root, port)
            except SystemExit:
                logger.warning("SDK backend failed, falling back to CLI backend")
                args.backend = "cli"

        if args.backend == "cli":
            server_process = start_server(args.wandb_root, port)

        try:
            _run_ready_server(args, port, server_process, server_container)

        except KeyboardInterrupt:
            logger.info("Interrupted - shutting down ...")

        finally:
            cleanup_resources(server_process, server_container)
            logger.info("Server stopped.")


if __name__ == "__main__":
    main()
