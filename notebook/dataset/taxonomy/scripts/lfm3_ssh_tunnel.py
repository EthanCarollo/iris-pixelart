#!/usr/bin/env python3
"""Private SSH forwarding for the remote vLLM API, with reconnect on transport loss."""
import argparse
from pathlib import Path
import select
import socketserver
import threading

import paramiko


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--ssh-port', type=int, default=20395)
    parser.add_argument('--user', default='eth')
    parser.add_argument('--password-file', type=Path, required=True)
    parser.add_argument('--local-port', type=int, default=8092)
    parser.add_argument('--remote-port', type=int, default=8092)
    args = parser.parse_args()
    lock, clients = threading.Lock(), []
    def transport():
        with lock:
            if clients and clients[-1].get_transport() and clients[-1].get_transport().is_active():
                return clients[-1].get_transport()
            for previous in clients:
                previous.close()
            clients.clear()
            client = paramiko.SSHClient()
            client.load_system_host_keys()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(args.host, port=args.ssh_port, username=args.user,
                           password=args.password_file.read_text().strip(), timeout=30,
                           look_for_keys=False, allow_agent=False)
            client.get_transport().set_keepalive(20)
            clients.append(client)
            return client.get_transport()
    class Forwarder(socketserver.BaseRequestHandler):
        def handle(self):
            try:
                channel = transport().open_channel('direct-tcpip', ('127.0.0.1', args.remote_port), self.client_address)
            except (OSError, paramiko.SSHException):
                return
            try:
                while True:
                    ready, _, _ = select.select([self.request, channel], [], [], 30)
                    if self.request in ready:
                        data = self.request.recv(262144)
                        if not data:
                            break
                        channel.sendall(data)
                    if channel in ready:
                        data = channel.recv(262144)
                        if not data:
                            break
                        self.request.sendall(data)
            except (OSError, paramiko.SSHException):
                pass
            finally:
                channel.close()
    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True
    transport()
    with Server(('127.0.0.1', args.local_port), Forwarder) as server:
        print(f'SSH tunnel ready: 127.0.0.1:{args.local_port} -> remote localhost:{args.remote_port}', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
