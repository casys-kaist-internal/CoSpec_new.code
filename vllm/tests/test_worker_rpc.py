"""Unit tests for CoSpec v2 worker RPC."""

import multiprocessing

import pytest

from vllm.cospec.worker_rpc import (
    DraftCommand,
    DraftResponse,
    DraftWorkerRPC,
    DraftWorkerServer,
    create_draft_worker_pipe,
)


class TestWorkerRPC:

    def test_create_pipe(self):
        parent_conn, child_conn = create_draft_worker_pipe()
        parent_conn.send("test")
        assert child_conn.recv() == "test"
        parent_conn.close()
        child_conn.close()

    def test_ping_pong(self):
        parent_conn, child_conn = create_draft_worker_pipe()

        # Simulate server in a thread
        import threading

        def server_loop():
            while True:
                cmd, kwargs = child_conn.recv()
                if cmd == DraftCommand.PING:
                    child_conn.send((DraftResponse.OK, "pong"))
                    break
                elif cmd == DraftCommand.SHUTDOWN:
                    child_conn.send((DraftResponse.OK, None))
                    break

        t = threading.Thread(target=server_loop)
        t.start()

        rpc = DraftWorkerRPC(parent_conn)
        assert rpc.ping() is True

        t.join(timeout=5)
        parent_conn.close()
        child_conn.close()

    def test_shutdown(self):
        parent_conn, child_conn = create_draft_worker_pipe()

        import threading

        def server_loop():
            cmd, kwargs = child_conn.recv()
            assert cmd == DraftCommand.SHUTDOWN
            child_conn.send((DraftResponse.OK, None))

        t = threading.Thread(target=server_loop)
        t.start()

        rpc = DraftWorkerRPC(parent_conn)
        rpc.shutdown()

        t.join(timeout=5)
        parent_conn.close()
        child_conn.close()

    def test_command_enum_values(self):
        assert DraftCommand.PROPOSE.value == "propose"
        assert DraftCommand.SET_PARTITION.value == "set_partition"
        assert DraftCommand.SHUTDOWN.value == "shutdown"
