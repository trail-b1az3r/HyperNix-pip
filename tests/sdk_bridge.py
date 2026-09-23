"""Route the SDK's urllib requests into a FastAPI TestClient.

The SDK and waiter talk HTTP through ``urllib``; the suite's servers are
TestClients. This is the ``opener`` the SDK's transport accepts, so a
test exercises the real client code against the real app with no socket.
"""
from __future__ import annotations

import io
import urllib.error
import urllib.parse
from email.message import Message


class _Response(io.BytesIO):
    def __init__(self, status: int, body: bytes, headers: dict[str, str]):
        super().__init__(body)
        self.status = status
        self.headers = headers

    def getcode(self) -> int:
        return self.status


def opener_for(test_client):
    def urlopen(request, timeout=None, **_kwargs):
        parts = urllib.parse.urlsplit(request.full_url)
        path = parts.path + (f"?{parts.query}" if parts.query else "")
        response = test_client.request(
            request.get_method(), path, content=request.data,
            headers=dict(request.header_items()),
        )
        headers = dict(response.headers)
        if response.status_code >= 400:
            message = Message()
            for key, value in headers.items():
                message[key] = value
            raise urllib.error.HTTPError(
                request.full_url, response.status_code, "error", message, io.BytesIO(response.content)
            )
        return _Response(response.status_code, response.content, headers)

    return urlopen


def sdk_client(test_client, credential=None, *, cls=None):
    from hypernix.t1sdk import HTTPTransport, T1Client

    cls = cls or T1Client
    transport = HTTPTransport("http://testserver", credential=credential,
                              opener=opener_for(test_client))
    return cls("http://testserver", credential=credential, transport=transport)
