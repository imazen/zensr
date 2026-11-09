"""Read-only local demo server with optional TLS and isolated WASM headers."""
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import ssl
from urllib.parse import unquote, urlsplit


def main(a):
    class Handler(SimpleHTTPRequestHandler):
        def translate_path(self, path):
            path=unquote(urlsplit(path).path)
            root=a.assets
            if path.startswith('/gallery/'):
                path=path.removeprefix('/gallery/')
                root=a.gallery
                if not (path.endswith('.png') or path=='witnesses.html'):
                    return str(root/'not-found')
            else:path=path.lstrip('/') or 'index.html'
            candidate=(root/path).resolve()
            if not candidate.is_relative_to(root.resolve()):return str(root/'not-found')
            return str(candidate)

        def end_headers(self):
            self.send_header('Cross-Origin-Opener-Policy','same-origin')
            self.send_header('Cross-Origin-Embedder-Policy','require-corp')
            self.send_header('Cross-Origin-Resource-Policy','same-origin')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Cache-Control','no-cache')
            super().end_headers()

        def list_directory(self, path):
            self.send_error(404)
            return None

    server=ThreadingHTTPServer(('0.0.0.0',a.port),Handler)
    if a.cert:
        context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(a.cert,a.key)
        server.socket=context.wrap_socket(server.socket,server_side=True)
    print(f'Listening on port {a.port}, TLS={bool(a.cert)}',flush=True)
    server.serve_forever()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--gallery',type=Path,required=True)
    p.add_argument('--port',type=int,default=3331)
    p.add_argument('--cert',type=Path)
    p.add_argument('--key',type=Path)
    a=p.parse_args()
    if not 3000<=a.port<=3999 or bool(a.cert)!=bool(a.key):p.error('invalid port or TLS pair')
    main(a)
