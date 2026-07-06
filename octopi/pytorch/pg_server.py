"""Ephemeral PostgreSQL server for multi-node Optuna model-explore.

SQLite cannot be shared safely by concurrent SLURM jobs running on different
nodes over a network filesystem (Lustre): WAL mode requires same-host shared
memory, and rollback-journal locking is both slow and corruption-prone across
nodes. Optuna's own guidance is to use a real server DB for multi-worker runs.

This module lets the model-explore *supervisor* stand up a private PostgreSQL
instance on its own node. Worker jobs on other nodes then connect over TCP via
the ``postgresql://<host>:<port>/<db>`` URL the supervisor hands them. The data
directory lives on shared storage so the DB survives a supervisor restart.

Only one process (this server) ever touches the data-dir files, from one node,
so there is no cross-node filesystem-locking problem — Postgres handles all
concurrency for the many worker connections.
"""

from __future__ import annotations

import atexit
import getpass
import os
import signal
import socket
import subprocess
import time

import psycopg2


def _find_free_port() -> int:
    """Pick a currently-free TCP port. Small TOCTOU race, acceptable here."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class PostgresServer:
    """Manage the lifecycle of a private PostgreSQL cluster for one search.

    Typical use (context manager)::

        with PostgresServer(data_dir=f"{output}/pgdata") as pg:
            storage_url = pg.url  # postgresql://user@host:port/optuna
            ... run workers ...
    """

    def __init__(
        self,
        data_dir: str,
        dbname: str = "optuna",
        port: int | None = None,
        host: str | None = None,
        user: str | None = None,
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.dbname = dbname
        self.user = user or getpass.getuser()
        # Hostname must be resolvable from the worker nodes; on SLURM clusters
        # the short node name (e.g. "cpu-b-6") is cluster-wide resolvable.
        self.host = host or socket.gethostname()
        self.port = port or _find_free_port()
        self.log_file = os.path.join(self.data_dir, "postgres.log")
        # Unix socket lives in a short, node-local, writable dir (NOT the Lustre
        # data dir, whose long path would exceed the ~107-char socket limit, and
        # NOT the default /var/run/postgresql, which users can't write to).
        # Admin connections use loopback TCP anyway; workers use TCP over the net.
        self.socket_dir = f"/tmp/pg_sock_{self.user}_{self.port}"
        self._started = False
        self._stopped = False

    # -- URLs -----------------------------------------------------------------
    @property
    def url(self) -> str:
        """SQLAlchemy/Optuna URL that worker nodes use to connect over TCP."""
        return f"postgresql://{self.user}@{self.host}:{self.port}/{self.dbname}"

    def _admin_dsn(self, dbname: str = "postgres") -> dict:
        """Local (loopback) connection params for admin tasks like CREATE DATABASE."""
        return dict(host="127.0.0.1", port=self.port, user=self.user, dbname=dbname)

    # -- lifecycle ------------------------------------------------------------
    def _initdb_if_needed(self) -> None:
        version_marker = os.path.join(self.data_dir, "PG_VERSION")
        if os.path.exists(version_marker):
            return  # cluster already initialized (persisted on shared storage)
        os.makedirs(self.data_dir, exist_ok=True)
        print(f"[postgres] initializing cluster at {self.data_dir}", flush=True)
        subprocess.run(
            ["initdb", "-D", self.data_dir, "-U", self.user,
             "--auth-local=trust", "--auth-host=trust", "--encoding=UTF8"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        # Allow worker nodes to connect over TCP. Data is non-sensitive Optuna
        # trial state on a private cluster network, so trust auth is acceptable.
        hba = os.path.join(self.data_dir, "pg_hba.conf")
        with open(hba, "a") as f:
            f.write("\n# Added by octopi PostgresServer: allow cluster nodes\n")
            f.write("host    all    all    0.0.0.0/0    trust\n")

    def _clear_stale_pidfile(self) -> None:
        """A hard-killed supervisor can leave postmaster.pid behind, blocking start."""
        if self.is_running():
            return
        pid_file = os.path.join(self.data_dir, "postmaster.pid")
        if os.path.exists(pid_file):
            print("[postgres] removing stale postmaster.pid", flush=True)
            os.remove(pid_file)

    def is_running(self) -> bool:
        r = subprocess.run(["pg_ctl", "-D", self.data_dir, "status"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return r.returncode == 0

    def _create_db_if_needed(self) -> None:
        # Wait for the postmaster to accept connections, then ensure the DB exists.
        last_err = None
        for _ in range(30):
            try:
                conn = psycopg2.connect(**self._admin_dsn(), connect_timeout=5)
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (self.dbname,))
                    if cur.fetchone() is None:
                        cur.execute(f'CREATE DATABASE "{self.dbname}"')
                        print(f"[postgres] created database {self.dbname!r}", flush=True)
                conn.close()
                return
            except psycopg2.OperationalError as e:
                last_err = e
                time.sleep(1)
        raise RuntimeError(f"Postgres did not become ready: {last_err}")

    def start(self) -> "PostgresServer":
        if self._started:
            return self
        self._initdb_if_needed()
        self._clear_stale_pidfile()
        if not self.is_running():
            os.makedirs(self.socket_dir, exist_ok=True)
            print(f"[postgres] starting on {self.host}:{self.port} "
                  f"(data={self.data_dir})", flush=True)
            subprocess.run(
                ["pg_ctl", "-D", self.data_dir, "-l", self.log_file, "-w",
                 "-o", f"-p {self.port} -c listen_addresses='*' "
                       f"-c unix_socket_directories={self.socket_dir}", "start"],
                check=True,
            )
        self._create_db_if_needed()
        self._started = True
        self._stopped = False
        # Defensive teardown so a crashing/terminated supervisor does not orphan
        # the postmaster. SIGKILL can't be caught; a stale pidfile is cleaned on
        # the next start() instead.
        atexit.register(self.stop)
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                prev = signal.getsignal(sig)
                signal.signal(sig, self._make_signal_handler(sig, prev))
            except (ValueError, OSError):
                pass  # not in main thread; atexit still covers normal exit
        print(f"[postgres] ready — storage url: {self.url}", flush=True)
        return self

    def _make_signal_handler(self, sig, prev):
        def _handler(signum, frame):
            self.stop()
            if callable(prev):
                prev(signum, frame)
            else:
                raise SystemExit(128 + signum)
        return _handler

    def stop(self) -> None:
        if self._stopped or not self._started:
            return
        self._stopped = True
        print("[postgres] stopping server", flush=True)
        subprocess.run(["pg_ctl", "-D", self.data_dir, "-m", "fast", "-w", "stop"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # -- context manager ------------------------------------------------------
    def __enter__(self) -> "PostgresServer":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
